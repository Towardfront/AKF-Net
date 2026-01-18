import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import ConvLSTMCell, BasicConv2d
from models.base.MHSA import MHSA
from models.lskblock import LSKblock
class FeatFusion(nn.Module):
    '''
    Dual-source Information Fusion (DSF) + Multi-scale Feature Fusion (MSF)
    '''
    def __init__(self, channel=64):
        super(FeatFusion, self).__init__()

        # DSF: y = gamma * x + beta
        self.beta_proj2 = nn.Linear(channel, channel+8)
        self.gamma_proj2 = nn.Linear(channel, channel+8)
        self.norm2 = nn.InstanceNorm2d(channel+8)

        self.beta_proj3 = nn.Linear(channel, channel+8)
        self.gamma_proj3 = nn.Linear(channel, channel+8)
        # 参数为num_features
        self.norm3 = nn.InstanceNorm2d(channel+8)

        self.beta_proj4 = nn.Linear(channel, channel+8)
        self.gamma_proj4 = nn.Linear(channel, channel+8)
        self.norm4 = nn.InstanceNorm2d(channel+8)

        self.fusion_process2 = BasicConv2d(channel+8, channel, 3, padding=1)
        self.fusion_process3 = BasicConv2d(channel+8, channel, 3, padding=1)
        self.fusion_process4 = BasicConv2d(channel+8, channel, 3, padding=1)

        # MSF: convlstm
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lstmcell43 = ConvLSTMCell(input_dim=channel, hidden_dim=channel, kernel_size=(3, 3), bias=True)
        # d(out)=d(in)*scale_factor
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lstmcell32 = ConvLSTMCell(input_dim=channel, hidden_dim=channel, kernel_size=(3, 3), bias=True)


    def forward(self, ref_x, x):
        x2, x3, x4 = x
        bs = ref_x.shape[0]

        # DSF
        coord_feat2 = self._make_coord(bs, x2.shape[2], x2.shape[3])
        coord_feat3 = self._make_coord(bs, x3.shape[2], x3.shape[3])
        coord_feat4 = self._make_coord(bs, x4.shape[2], x4.shape[3])
        if ref_x.is_cuda:
            coord_feat2 = coord_feat2.cuda()
            coord_feat3 = coord_feat3.cuda()
            coord_feat4 = coord_feat4.cuda()

        x2 = self.norm2(torch.cat([x2, coord_feat2], 1))
        x3 = self.norm3(torch.cat([x3, coord_feat3], 1))
        x4 = self.norm4(torch.cat([x4, coord_feat4], 1))
        
        # y = gamma * x + beta
        beta2 = torch.tanh(self.beta_proj2(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x2)
        gamma2 = torch.tanh(self.gamma_proj2(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x2)
        beta3 = torch.tanh(self.beta_proj3(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x3)
        gamma3 = torch.tanh(self.gamma_proj3(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x3)
        beta4 = torch.tanh(self.beta_proj4(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x4)
        gamma4 = torch.tanh(self.gamma_proj4(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x4)

        x2 = self.fusion_process2(F.relu(gamma2 * x2 + beta2))
        x3 = self.fusion_process3(F.relu(gamma3 * x3 + beta3))
        x4 = self.fusion_process4(F.relu(gamma4 * x4 + beta4))   

        # MSF
        x4_h, x4_c = self.upsample4(x4), self.upsample4(x4)
        x3_h, x3_c = self.lstmcell43(input_tensor=x3, cur_state=[x4_h, x4_c])
        # print('x3: ', x3_h.shape)

        x3_h, x3_c = self.upsample3(x3_h), self.upsample3(x3_c)
        x2_h, x2_c = self.lstmcell32(input_tensor=x2, cur_state=[x3_h, x3_c])
        # print('x2: ', x2_h.shape)

        return x2_h

    def _make_coord(self, batch, height, width):
        xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
        xv_min = (xv.float()*2 - width)/width
        yv_min = (yv.float()*2 - height)/height
        xv_max = ((xv+1).float()*2 - width)/width
        yv_max = ((yv+1).float()*2 - height)/height
        xv_ctr = (xv_min+xv_max)/2
        yv_ctr = (yv_min+yv_max)/2
        hmap = torch.ones(height, width)*(1./height)
        wmap = torch.ones(height, width)*(1./width)
        coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
            xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
            xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
            hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
        coord = coord.unsqueeze(0).repeat(batch,1,1,1)
        return coord


class RFE(nn.Module):
    ''' 
    Referring Feature Enrichment (RFE) Module
    Follow implementation of https://github.com/dvlab-research/PFENet/blob/master/model/PFENet.py
    '''
    def __init__(self, d_model=64):
        super(RFE, self).__init__()

        self.d_model = d_model
        self.pyramid_bins = [44, 22, 11]        # 352 // 8, 352 // 16, 352 // 32

        self.avgpool_list = [nn.AdaptiveAvgPool2d(bin_) for bin_ in self.pyramid_bins if bin_ > 1]

        self.init_merge = []
        self.alpha_conv = []
        self.beta_conv = []
        self.inner_cls = []

        for idx in range(len(self.pyramid_bins)):
            if idx > 0:
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(self.d_model*2, self.d_model, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.d_model),
                    nn.ReLU()
                )) 
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(self.d_model + 1, self.d_model, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(self.d_model, 1, kernel_size=1)
            )) 
       
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.pyramid_cat_conv = nn.Sequential(
            nn.Conv2d(self.d_model*len(self.pyramid_bins), self.d_model, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),                          
        )              
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),   
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),                             
        )

        # self.lsk = LSKblock(dim=self.d_model)

    def forward(self, feats, mask):
        '''
        feats: [bs, 64, 44, 44]
        sf: [bs, 1, 44, 44]
        '''

        inner_out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin_ = int(feats.shape[2] * tmp_bin)
                feats_bin = nn.AdaptiveAvgPool2d(bin)(feats)
            else:
                bin_ = tmp_bin
                feats_bin = self.avgpool_list[idx](feats)
            
            mask_bin = F.interpolate(mask, size=(bin_, bin_), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([feats_bin, mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin_, bin_), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin

            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(feats.size(2), feats.size(3)), mode='bilinear', align_corners=True)
            
            inner_out_list.append(inner_out_bin)
            pyramid_feat_list.append(merge_feat_bin)
            
        feats_refine = self.pyramid_cat_conv(torch.cat(pyramid_feat_list, 1))
        feats_refine = self.conv_block(feats_refine) + feats_refine
        # feats_refine=self.lsk(feats_refine)


        return feats_refine, inner_out_list




#######################################################################################################################



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale



class FEM(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.Dropout(0.2)
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.Dropout(0.2)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        #
        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

    def forward(self, q, s):
        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication

        # common feature learning
        v_q = self.common_v(q).view(batch_size, self.inter_channels, -1)
        v_q = v_q.permute(0, 2, 1)

        v_s = self.common_v(s).view(batch_size, self.inter_channels, -1)
        v_s = v_s.permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1)
        k_x = k_x.permute(0, 2, 1)

        q_x = self.query(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        p_s = self.Trans_s(p_s)
        # Intra-image channel attention
        E_s = self.ChannelGate(s) * p_s
        E_s = E_s + s
        E_s=self.dropout(E_s)

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        q_s = self.Trans_q(q_s)
        # Intra-image channel attention
        E_q = self.ChannelGate(q) * q_s
        E_q = E_q + q
        E_q=self.dropout(E_q)

        return E_q, E_s


class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

class SE_ASPP(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.senet=SE_Block(in_planes=dim_out*5)
        self.MHSA=MHSA(n_dims=dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # print(conv1x1.shape, conv3x3_1.shape, conv3x3_2.shape, conv3x3_3.shape, global_feature.shape)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             #加入通道注意力机制
        MHSA=self.MHSA(seaspp1)

        se_feature_cat=MHSA*feature_cat
        result = self.conv_cat(se_feature_cat)

        return result



############################################################################################

import torch
import torch.nn as nn


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

