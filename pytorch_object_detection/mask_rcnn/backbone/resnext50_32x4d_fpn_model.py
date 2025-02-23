import os
import torch.nn as nn
import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops.misc import FrozenBatchNorm2d

def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


# ------------------------------
# 自定义 ResNeXt50_32x4d Backbone
# ------------------------------

def resnext50_32x4d_fpn_backbone(
    weights: str = None,
    norm_layer=torch.nn.BatchNorm2d,
    trainable_layers: int = 3,
    returned_layers=None,
    extra_blocks=None,
    **kwargs
):
    """
    构建 ResNeXt50_32x4d + FPN 的 Backbone
    参数：
        pretrain_path (str): 预训练权重路径（ImageNet 预训练的 ResNeXt50_32x4d）
        trainable_layers (int): 可训练层数（从最后一层开始计数，0~5）
    """
    # 创建 ResNeXt50_32x4d 模型（替换 ResNet）
    backbone = torchvision.models.resnext50_32x4d(weights=weights,
                                                  norm_layer=norm_layer)
    
    backbone.avgpool = nn.Identity() 
    backbone.fc = nn.Identity()
    
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(backbone, 0.0)
    
    # 冻结指定层（默认训练最后 3 层）
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
        
    # 定义 FPN
    # 返回的通道数需要与 ResNeXt 的输出匹配（假设与 ResNet 相同）
    if returned_layers is None:
        return_layers = {
            "layer1": "0",  # stride 4
            "layer2": "1",  # stride 8
            "layer3": "2",  # stride 16
            "layer4": "3",  # stride 32
        }
    
    # 创建 FPN（输入通道数需与 ResNeXt 的输出匹配）
    in_channels_stage2 = backbone.layer1[-1].conv3.out_channels
    in_channels_list = [
        in_channels_stage2,            # layer1 的输出通道
        in_channels_stage2 * 2,        # layer2 的输出通道
        in_channels_stage2 * 4,        # layer3 的输出通道
        in_channels_stage2 * 8,        # layer4 的输出通道
    ]
    out_channels = 256  # FPN 输出统一为 256 通道
    
    # 组合为 FPN Backbone
    return torchvision.models.detection.backbone_utils.BackboneWithFPN(
        backbone=backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks,
        **kwargs
    )

