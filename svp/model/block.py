import torch
from functools import partial
from torch import Tensor
import torch.nn as nn
from typing import Optional

from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,  # 模型的维度
        ssm_cfg=None,  # 随机状态模块（SSM）的配置，默认为None
        norm_epsilon=1e-5,  # 层归一化中的epsilon值，防止除以零
        drop_path=0.,  # 路径dropout的比例
        rms_norm=False,  # 是否使用RMSNorm代替LayerNorm
        residual_in_fp32=False,  # 是否在FP32中使用残差连接
        fused_add_norm=False,  # 是否使用融合的加法和归一化操作
        layer_idx=None,  # 层的索引，默认为None
        device=None,  # 操作设备，默认为None
        dtype=None,  # 数据类型，默认为None
        if_bimamba=False,  # 是否使用Bimamba结构
        bimamba_type="none",  # Bimamba的类型，默认为"none"
        if_devide_out=False,  # 是否在Mixer中使用devide out
        init_layer_scale=None,  # 层初始化的比例因子，默认为None
):
    """
    创建一个模型块，可以配置不同的归一化类型、dropout和混合器类。

    参数:
    - d_model: 模型的维度。
    - ssm_cfg: 随机状态模块（SSM）的配置，默认为空字典。
    - norm_epsilon: 层归一化中的epsilon值，防止除以零。
    - drop_path: 路径dropout的比例。
    - rms_norm: 是否使用RMSNorm代替LayerNorm。
    - residual_in_fp32: 是否在FP32中使用残差连接。
    - fused_add_norm: 是否使用融合的加法和归一化操作。
    - layer_idx: 层的索引，默认为None。
    - device: 操作设备，默认为None。
    - dtype: 数据类型，默认为None。
    - if_bimamba: 是否使用Bimamba结构。
    - bimamba_type: Bimamba的类型，默认为"none"。
    - if_devide_out: 是否在Mixer中使用devide out。
    - init_layer_scale: 层初始化的比例因子，默认为None。

    返回:
    - block: 配置好的模型块。
    """
    if if_bimamba:
        bimamba_type = "v1"  # 默认Bimamba类型为v1
    if ssm_cfg is None:
        ssm_cfg = {}  # 如果没有提供SSM配置，则默认为空字典
    # 准备用于创建Mixer和归一化类的工厂参数
    factory_kwargs = {"device": device, "dtype": dtype}
    # 根据配置选择Mixer类
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    # 根据是否使用RMSNorm选择归一化类
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # 创建模型块
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx  # 设置层索引
    return block