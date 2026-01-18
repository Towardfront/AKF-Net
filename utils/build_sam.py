# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

import logging
import torch
from hydra import compose, initialize, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
):

    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    # # ✅ 自动找到 config 文件所在目录
    # config_dir = os.path.dirname(config_file)
    # config_name = os.path.basename(config_file).replace(".yaml", "")

    # # ✅ 手动初始化 Hydra（最关键）
    # with initialize_config_dir(config_dir=config_dir, version_base=None):
    #     cfg = compose(config_name=config_name, overrides=hydra_overrides_extra)

    config_path = os.path.abspath(config_file)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace(".yaml", "")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides_extra)

    OmegaConf.resolve(cfg)

    # ✅ 构建模型
    model = instantiate(cfg.model, _recursive_=True)

    # ✅ 加载权重
    _load_checkpoint(model, ckpt_path)

    model = model.to(device)
    if mode == "eval":
        model.eval()

    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
):

    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]

    hydra_overrides.extend(hydra_overrides_extra)

    config_dir = os.path.dirname(config_file)
    config_name = os.path.basename(config_file).replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)

    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)

    model = model.to(device)
    if mode == "eval":
        model.eval()

    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint successfully")

# import logging

# import torch
# from hydra import compose, initialize
# from hydra.utils import instantiate
# from omegaconf import OmegaConf


# def build_sam2(
#     config_file,
#     ckpt_path=None,
#     device="cuda",
#     mode="eval",
#     hydra_overrides_extra=[],
#    