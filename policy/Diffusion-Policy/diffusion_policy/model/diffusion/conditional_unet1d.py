from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )


        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, sample_long, cond):
        '''
            sample_long : [ batch_size sample_long in_channels sample_long horizon ]
            cond : [ batch_size sample_long cond_dim]

            returns:
            out : [ batch_size sample_long out_channels sample_long horizon ]
        '''
        out = self.blocks[0](sample_long)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(sample_long)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        cross_attention,
        num_ctrl_pts=8,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        self.cross_attention = cross_attention
        self.num_ctrl_pts = num_ctrl_pts
        self.depth = len(down_dims)

        # learnable CLS token list (for upsample and downsample)
        self.cls_token_list = nn.ModuleList([
            nn.Parameter(torch.randn(1, 3, down_dims[0]*self.num_ctrl_pts//2))  # 这里把这个维度给写死了，默认了最后一层的 time_horizon 为 1
            for dim in down_dims
        ])

        # learnable CLS token (for mid process)
        self.mid_cls_tokens = nn.Parameter(torch.randn(1, 3, down_dims[0]*self.num_ctrl_pts//2))

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample_long: torch.Tensor, 
            sample_mid: torch.Tensor, 
            sample_short: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond_long=None, 
            global_cond_mid=None, global_cond_short=None, **kwargs):
        """
        sample_long: (B,T,input_dim)
        sample_mid: (B,T,input_dim)
        sample_short: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample_long = einops.rearrange(sample_long, 'b h t -> b t h')
        sample_mid = einops.rearrange(sample_mid, 'b h t -> b t h')
        sample_short = einops.rearrange(sample_short, 'b h t -> b t h')
        
        batch_size = sample_long.shape[0]

        # 1. 时间步编码，并把时间步编码和 obs 拼接
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample_long.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample_long.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample_long.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond_long is not None:
            global_feature_long = torch.cat([
                global_feature, global_cond_long
            ], axis=-1)

        if global_cond_mid is not None:
            global_feature_mid = torch.cat([
                global_feature, global_cond_mid
            ], axis=-1)
        
        if global_cond_short is not None:
            global_feature_short = torch.cat([
                global_feature, global_cond_short
            ], axis=-1)
        
        # encode local features
        # 这种情况不会出现，local_cond 是 None 已经被写死了，下面的代码没用，就不改了
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            sample_long = resnet(local_cond, global_feature_long)
            h_local.append(sample_long)
            sample_long = resnet2(local_cond, global_feature_long)
            h_local.append(sample_long)
        

        # 开三个数组保存中间特征，便于 skip conn
        h_long = []
        h_mid = []
        h_short = []

        # 下采样过程
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            sample_long = resnet(sample_long, global_feature_long)
            sample_mid = resnet(sample_mid, global_feature_mid)
            sample_short = resnet(sample_short, global_feature_short)

            if idx == 0 and len(h_local) > 0:
                sample_long = sample_long + h_local[0]
                sample_mid = sample_mid + h_local[0]
                sample_short = sample_short + h_local[0]

            sample_long = resnet2(sample_long, global_feature_long)
            sample_mid = resnet2(sample_mid, global_feature_mid)
            sample_short = resnet2(sample_short, global_feature_short)

            h_long.append(sample_long)
            h_mid.append(sample_mid)
            h_short.append(sample_short)

            # max_pool 时间尺寸减半
            sample_long = downsample(sample_long)
            sample_mid = downsample(sample_mid)
            sample_short = downsample(sample_short)
                        
            _, C, T = sample_long.shape 

            sample_long = sample_long.reshape(batch_size, -1) # (B, C*T)
            sample_mid = sample_mid.reshape(batch_size, -1)
            sample_short = sample_short.reshape(batch_size, -1)

            tokens_from_sample = torch.stack([sample_long, sample_mid, sample_short], dim=1) # (B, 3, C*T)
            cls_tokens = self.cls_token_list[idx].expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens_from_sample], dim=1) # (B, 6, C*T)

            output = self.cross_attention(tokens) # (B, 6, C*T)

            sample_long = output[:, 0, :].reshape(batch_size, C, T)
            sample_mid = output[:, 1, :].reshape(batch_size, C, T)
            sample_short = output[:, 2, :].reshape(batch_size, C, T)

        # 中间层，实际只有 1 块，2 个 resnet
        for mid_module in self.mid_modules:

            # sample_long/mid/short 的形状：(B, C, T)
            sample_long = mid_module(sample_long, global_feature_long)
            sample_mid = mid_module(sample_mid, global_feature_mid)
            sample_short = mid_module(sample_short, global_feature_short)

            _, C, T = sample_long.shape

            sample_long = sample_long.reshape(batch_size, -1)
            sample_mid = sample_mid.reshape(batch_size, -1)
            sample_short = sample_short.reshape(batch_size, -1)

            tokens_from_sample = torch.stack([sample_long, sample_mid, sample_short], dim = 1) #(batch_size, 3， C*T)
            cls_tokens = self.mid_cls_tokens.expand(batch_size, -1, -1) #(B, 3, C*T)
            tokens = torch.cat([cls_tokens, tokens_from_sample], dim=1)

            output = self.cross_attention(tokens) # (B, 6, C*T)

            # 取出前三个输出
            sample_long = output[:, 0, :].reshape(batch_size, C, T)
            sample_mid = output[:, 1, :].reshape(batch_size, C, T)
            sample_short = output[:, 2, :].reshape(batch_size, C, T)


        # 上采样过程 
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            sample_long = torch.cat((sample_long, h_long.pop()), dim=1)
            sample_mid = torch.cat((sample_mid, h_mid.pop()), dim=1)
            sample_short = torch.cat((sample_short, h_short.pop()), dim=1)

            sample_long = resnet(sample_long, global_feature_long)
            sample_mid = resnet(sample_mid, global_feature_mid)
            sample_short = resnet(sample_short, global_feature_short)

            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                sample_long = sample_long + h_local[1]
                sample_mid = sample_mid + h_local[1]
                sample_short = sample_short + h_local[1]

            sample_long = resnet2(sample_long, global_feature_long)
            sample_mid = resnet2(sample_mid, global_feature_mid)
            sample_short = resnet2(sample_short, global_feature_short)

            sample_long = upsample(sample_long)
            sample_mid = upsample(sample_mid)
            sample_short = upsample(sample_short)
            
            _, C, T = sample_long.shape 

            sample_long = sample_long.reshape(batch_size, -1) # (B, C*T)
            sample_mid = sample_mid.reshape(batch_size, -1)
            sample_short = sample_short.reshape(batch_size, -1)

            tokens_from_sample = torch.stack([sample_long, sample_mid, sample_short], dim=1) # (B, 3, C*T)
            cls_tokens = self.cls_token_list[self.depth - idx - 1].expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens_from_sample], dim=1) # (B, 6, C*T)

            output = self.cross_attention(tokens) # (B, 6, C*T)

            sample_long = output[:, 0, :].reshape(batch_size, C, T)
            sample_mid = output[:, 1, :].reshape(batch_size, C, T)
            sample_short = output[:, 2, :].reshape(batch_size, C, T)

        sample_long = self.final_conv(sample_long)
        sample_mid = self.final_conv(sample_mid)
        sample_short = self.final_conv(sample_short)

        sample_long = einops.rearrange(sample_long, 'b t h -> b h t')
        sample_mid = einops.rearrange(sample_mid, 'b t h -> b h t')
        sample_short = einops.rearrange(sample_short, 'b t h -> b h t')

        return sample_long, sample_mid, sample_short

