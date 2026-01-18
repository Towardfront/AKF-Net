import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.utils import BasicConv2d
from models.modules import FeatFusion, RFE,FEM,SE_ASPP,LSKblock
from models.base.swin_transformer import SwinTransformer
from models.lskblock import LSKblock
from models.SAM2UNeXT import SAM2UNeXT


class Network(nn.Module):
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.x2_down_channel = BasicConv2d(256, channel, 1)
        self.x3_down_channel = BasicConv2d(512, channel, 1)
        self.x4_down_channel = BasicConv2d(1024, channel, 1)
        self.reduce1 = nn.Conv2d(288 + 1024, 64, 1)
        self.reduce2 = nn.Conv2d(576 + 1024, 64, 1)
        self.reduce3 = nn.Conv2d(1152 + 1024, 64, 1)
        self.ref_proj = BasicConv2d(2048, channel, 1)
        self.feat_fusion = FeatFusion(channel=channel)
        self.relevance_norm = nn.BatchNorm2d(1)
        self.relevance_acti = nn.LeakyReLU(0.1, inplace=True)
        self.rfe = RFE(d_model=channel)
        self.cls = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.1), 
            nn.Conv2d(channel, 1, 1)
        )
        self.aspp=SE_ASPP(dim_in=channel,dim_out=channel)

        self.nlayers = [2, 2, 18, 2]
        
        self.lsk=LSKblock(dim=64)
        self.backbone = SAM2UNeXT('../SAM2-UNeXT-main/ckpt/model.safetensors')

    def forward(self, x, ref_x):
        bs, _, H, W = x.shape
        
        _, sam_o1, sam_o2, sam_o3 = self.backbone.sam(x)
        dino_o1, dino_o2, dino_o3 = self.backbone.dino(F.interpolate(x, size=(448, 448), mode='bilinear'))


        dino_o1 = F.interpolate(dino_o1, size=sam_o1.shape[-2:], mode='bilinear', align_corners=False)
        dino_o2 = F.interpolate(dino_o2, size=sam_o2.shape[-2:], mode='bilinear', align_corners=False)
        dino_o3 = F.interpolate(dino_o3, size=sam_o3.shape[-2:], mode='bilinear', align_corners=False)

        x2 = torch.cat([sam_o1, dino_o1], dim=1)   # (B, 288 + 1024, 48, 48)
        x3 = torch.cat([sam_o2, dino_o2], dim=1)   # (B, 576 + 1024, 24, 24)
        x4 = torch.cat([sam_o3, dino_o3], dim=1)   # (B, 1152 + 1024, 12, 12)

        x2 = self.reduce1(x2)
        x3 = self.reduce2(x3)
        x4 = self.reduce3(x4)

        ref_x = self.ref_proj(ref_x)

        x2_h = self.feat_fusion(ref_x=ref_x, x=[x2, x3, x4])

        x2_h=self.aspp(x2_h)
        out=self.lsk(x2_h)

        # Target Matching
        mask = torch.cat([F.conv2d(out[i].unsqueeze(0), ref_x[i].unsqueeze(0)) for i in range(bs)], 0)
        mask = self.relevance_acti(self.relevance_norm(mask))

        out, inner_out_list = self.rfe(out, mask)
        
        # Conv Head
        S_g = self.cls(out)

        S_g_pred = F.interpolate(S_g, size=(H, W), mode='bilinear', align_corners=True)      # (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        S_inner_preds = [F.interpolate(inner_out, size=(H, W), mode='bilinear', align_corners=True) for inner_out in inner_out_list]

        return S_g_pred, S_inner_preds
