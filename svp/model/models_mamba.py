# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
from torch import Tensor
from typing import Optional

from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from .patch_embed import PatchEmbed
from .block import create_block
from torch.nn import LayerNorm

from .rope import *
import random
# from ..dataset.datasets import build_dataset

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# __all__ = [
#     'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
#     'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
# ]


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    """
    初始化模型权重。

    参数:
    - m: 模型模块，可以是nn.Linear, nn.Conv2d, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d等类型的模块。

    说明:
    - 对于nn.Linear类型的模块，使用截断正态分布初始化权重，标准差为0.02，并将偏置初始化为0。
    - 对于nn.Conv2d类型的模块，使用lecun_normal_初始化权重，并将偏置初始化为0。
    - 对于归一化层（nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d），将偏置初始化为0，权重初始化为1。
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)  # 使用截断正态分布初始化权重
        if m.bias is not None:  # 确保偏置不为None时才初始化
            nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)  # 使用lecun_normal_初始化权重
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 将偏置初始化为0
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)  # 将归一化层的偏置初始化为0
        nn.init.ones_(m.weight)  # 将归一化层的权重初始化为1


def swish(x, beta):
    return x * torch.sigmoid(x * beta)

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-3):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class VisionMamba(nn.Module):
    """
    VisionMamba模型，一个用于视觉任务的Transformer模型。

    参数:
    - img_size: 输入图像的大小，默认为224。
    - patch_size: 图像切割成补丁的大小，默认为16。
    - stride: 图像切割的步长，默认为16。
    - depth: Transformer层数，默认为24。
    - embed_dim: 嵌入维度，默认为192。
    - channels: 图像通道数，默认为3。
    - num_classes: 分类类别数量，默认为1000。
    - ssm_cfg: 状态空间模型的配置，默认为None。
    - drop_rate: 丢弃率，默认为0.0。
    - drop_path_rate: 路径丢弃率，默认为0.1。
    - norm_epsilon: 归一化中的epsilon值，默认为1e-5。
    - rms_norm: 是否使用RMSNorm，默认为False。
    - initializer_cfg: 初始化配置，默认为None。
    - fused_add_norm: 是否使用融合的加法和归一化，默认为False。
    - residual_in_fp32: 是否在FP32中使用残差，默认为False。
    - device: 设备，默认为None。
    - dtype: 数据类型，默认为None。
    - pt_hw_seq_len: 点云序列的长度，默认为14。
    - if_bidirectional: 是否为双向，默认为False。
    - if_abs_pos_embed: 是否使用绝对位置嵌入，默认为False。
    - flip_img_sequences_ratio: 翻转图像序列的比例，默认为-1.0。
    - if_bimamba: 是否使用Bimamba，默认为False。
    - bimamba_type: Bimamba的类型，默认为'none'。
    - if_devide_out: 是否使用DivideOut，默认为False。
    - init_layer_scale: 层初始化比例，默认为None。
    """
    norm_f: LayerNorm

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,
                 embed_dim=192,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 if_abs_pos_embed=False,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 transfer_type="prompt",
                 prompt_type="addv4",
                 prompt_shared=False,
                 prompt_add_gen="mlp384*384",
                 prompt_depth=24,  # <=depth
                 prompt_dropout=0.0,
                 shared_layers=1,
                 dataset=None,
                 input_relu=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # 更新kwargs以包含factory_kwargs
        kwargs.update(factory_kwargs)
        super().__init__()

        # 初始化模型配置
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.if_abs_pos_embed = if_abs_pos_embed
        self.num_tokens = 1
        self.transfer_type = transfer_type
        self.input_relu = input_relu
        if self.transfer_type == "prompt":
            self.prompt_type = prompt_type
            self.prompt_shared = prompt_shared
            self.prompt_add_gen = prompt_add_gen
            self.prompt_dropout = prompt_dropout
            self.prompt_depth = prompt_depth
            self.shared_layers = shared_layers
            
        # 预训练参数
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # 为了与其他模型保持一致，使用num_features

        # 图像切割嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches  #196
        self.simam = simam_module()

        # 初始化位置嵌入和分类令牌
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        # 输出层

        if self.transfer_type == "prompt" and self.prompt_type == "addv4" and (dataset == "CUB200" or dataset == "NABIRDS" or dataset == "FLOWERS"):
            self.head = nn.Sequential(
                nn.Linear(self.num_features, self.num_features // 2),
                nn.Linear(self.num_features // 2, num_classes) if num_classes > 0 else nn.Identity()
            )
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 路径丢弃率的分段设置
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Mamba编码器层
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_devide_out=if_devide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        # 输出归一化
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # 应用初始化权重
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # 应用特定的权重初始化
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        if self.transfer_type != "none":
            self.init_prompt(patch_size, depth)

    def init_prompt(self, patch_size, depth):
        if self.transfer_type == "prompt":
            for k, p in self.norm_f.named_parameters():
                    p.requires_grad = False
            for k, p in self.layers.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

            for k, p in self.patch_embed.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False
            self.pos_embed.requires_grad = False
            self.cls_token.requires_grad = False

            # 提示的dropout
            self.prompt_dropout = nn.Dropout(self.prompt_dropout)
            # if project the prompt embeddings

            if self.prompt_type == "addv4":
                tmp = depth // self.shared_layers
                self.prompt_scale = nn.Parameter(torch.zeros([24, self.embed_dim]))
                self.prompt_shared_scale = nn.Parameter(torch.zeros([tmp, self.embed_dim]))
                self.prompt_generator_shared = nn.ModuleList(
                    nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim),
                        nn.SiLU(),
                    )
                    for i in range(depth//self.shared_layers)
                )
                if self.prompt_add_gen == "mlp384*32":
                    self.prompt_generator = nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(self.embed_dim, 32),
                            nn.Linear(64, self.embed_dim),
                            nn.SiLU(),
                        )
                        for i in range(depth)
                    )
                elif self.prompt_add_gen == "mlp384*64":
                    self.prompt_generator = nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(self.embed_dim, 64),
                            nn.Linear(64, self.embed_dim),
                            nn.SiLU(),
                        )
                        for i in range(depth)
                    )
                elif self.prompt_add_gen == "mlp384*128":
                    self.prompt_generator = nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(self.embed_dim, 128),
                            nn.Linear(128, self.embed_dim),
                            nn.SiLU(),
                        )
                        for i in range(depth)
                    )
                else:
                    self.prompt_generator = nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(self.embed_dim, 64),
                            nn.Linear(64, self.embed_dim),
                            nn.SiLU(),
                        )
                        for i in range(depth)
                    )
            else:
                # 不支持的prompt方式会抛出异常
                raise ValueError("Other prompt_type is not supported")



    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_token_rank=False):
        """
        Forward pass for the feature extraction part of the model.

        Args:
            x (Tensor): Input tensor to be processed.
            inference_params (dict, optional): Parameters for inference mode. Defaults to None.
            if_random_token_rank (bool, optional): Whether to randomly shuffle the token order. Defaults to False.

        Returns:
            Tensor: Processed tensor output from the feature extraction layers.
        """
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, num_patches:196, embed_dim:192]

        # Add class token
        token_position, x = self.add_clstoken(x)
        B, M, _ = x.shape  # 获取批次大小、patch数量、嵌入维度

        # Add positional embeddings
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # Add Prompt to the input sequence
        if self.transfer_type == "prompt":
            x, token_position = self.incorporate_prompt(x, token_position)

        # Main layer processing
        residual = None
        hidden_states = x #[batch_size, num_patches:197, embed_dim:192]
        M = x.shape[1]

        current_depth = 0
        for layer in self.layers:   # 24
            # Apply layer operations, including potential flipping and rope transformations

            # 🌟🌟🌟
            if self.transfer_type != "prompt" or current_depth == 0:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
            elif self.transfer_type == "prompt" and self.prompt_type == "addv4":
                hidden_states = self.incorporate_deep_prompt(hidden_states, token_position, current_depth)
                # 应用当前层的编码器模块
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )


            current_depth += 1


        # Final normalization and output preparation
        if not self.fused_add_norm:    #self.fused_add_norm=True
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:    # 🌟🌟
            # Fused add and norm operation
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # Return the appropriate output based on the model configuration
        return hidden_states[:, token_position, :], hidden_states


    def add_clstoken(self, x):
        B, M, _ = x.shape
        
        # Add class token in the middle
        cls_token = self.cls_token.expand(B, -1, -1)  # [batch_size, 1, embed_dim:192]
        token_position = M // 2
        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1)  # [batch_size, num_patches:197, embed_dim:192]

        M = x.shape[1]  # 更新patch数量为196+1
        return token_position, x

    def forward(self, x, return_features=False, inference_params=None, if_random_token_rank=False):
        if self.input_relu:
            x = self.simam(x)
        x, hidden_states = self.forward_features(x, inference_params, if_random_token_rank=if_random_token_rank)
        if return_features:
            return x
        x = self.head(x)
        return x, hidden_states

    def forward_wo_head(self, x, return_features=False, inference_params=None, if_random_token_rank=False):
        x, hidden_states = self.forward_features(x, inference_params, if_random_token_rank=if_random_token_rank)
        if return_features:
            return x
        return x, hidden_states

    def incorporate_prompt(self, x, token_position):
        """
        将提示嵌入与图像块嵌入相结合。

        参数:
        - x: 输入的图像嵌入数据，其形状为(batch_size, n_patches, embedding_dim)，
             其中batch_size是批次大小，n_patches是图像块的数量，embedding_dim是嵌入维度。

        返回值:
        - 经过结合提示嵌入后的数据，其形状为(batch_size, cls_token + n_prompt + n_patches, hidden_dim)，
          其中cls_token是分类token的数量，n_prompt是提示的数量，hidden_dim是隐藏层的维度。
        """
        # 计算批次大小
        B = x.shape[0]

        # 将提示嵌入扩展至与批次大小相匹配，并与图像块嵌入拼接
        if self.prompt_type == "addv4":
            x[:, :token_position, :] = x[:, :token_position, :] + self.prompt_dropout(
                self.prompt_generator[0](x[:, :token_position, :]))*self.prompt_scale[0] + self.prompt_dropout(
                self.prompt_generator_shared[0](x[:, :token_position, :]))*self.prompt_shared_scale[0]
            x[:, token_position + 1:, :] = x[:, token_position + 1:, :] + self.prompt_dropout(
                self.prompt_generator[0](x[:, token_position + 1:, :]))*self.prompt_scale[0] + self.prompt_dropout(
                self.prompt_generator_shared[0](x[:, :token_position, :]))*self.prompt_shared_scale[0]

        return x, token_position

    def incorporate_deep_prompt(self, hidden_states, token_position, current_depth):

        if self.prompt_type == "addv4":
            current_depth_tmp = current_depth
            hidden_states[:, :token_position, :] = hidden_states[:, :token_position, :] + self.prompt_dropout(
                self.prompt_generator[current_depth_tmp](hidden_states[:, :token_position, :]))*self.prompt_scale[current_depth_tmp] + self.prompt_dropout(
                self.prompt_generator_shared[current_depth_tmp//self.shared_layers](hidden_states[:, :token_position, :]))*self.prompt_shared_scale[current_depth_tmp//self.shared_layers]
            hidden_states[:, token_position + 1:, :] = hidden_states[:, token_position + 1:, :] + self.prompt_dropout(
                self.prompt_generator[current_depth_tmp](hidden_states[:, token_position + 1:, :]))*self.prompt_scale[current_depth_tmp] + self.prompt_dropout(
                self.prompt_generator_shared[current_depth_tmp//self.shared_layers](hidden_states[:, :token_position, :]))*self.prompt_shared_scale[current_depth_tmp//self.shared_layers]

        return hidden_states


