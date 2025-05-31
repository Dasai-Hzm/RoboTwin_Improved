from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import BertModel, BertConfig

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.bezier_curve.data_fit_bezier_curve import BezierFitter
from diffusion_policy.model.bezier_curve.upsample_bezier_curve import bezier_upsample

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            transformer_emb_size=512,
            num_ctrl_pts=8,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        
        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps


        config = BertConfig(hidden_size=transformer_emb_size, num_attention_heads=8, intermediate_size=transformer_emb_size * 4, num_hidden_layers=4)
   
        self.cross_attention = BertModel(config)
        
        model = ConditionalUnet1D(
            input_dim=input_dim,
            cross_attention=self.cross_attention,
            num_ctrl_pts=self.num_ctrl_pts,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
     


        self.obs_encoder = obs_encoder

        self.model = model

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        ###################################################################################
        ## 贝塞尔控制点拟合器
        self.num_ctrl_pts = num_ctrl_pts
        self.data_fitter_long = BezierFitter(input_dim=self.action_dim, num_control_points=self.num_ctrl_pts, horizon_length=self.horizon)
        self.data_fitter_mid = BezierFitter(input_dim=self.action_dim, num_control_points=self.num_ctrl_pts, horizon_length=self.horizon//2)
        self.data_fitter_short = BezierFitter(input_dim=self.action_dim, num_control_points=self.num_ctrl_pts, horizon_length=self.horizon//4)
        ###################################################################################

    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        
        # 这里还得改，已经没有 model_long 了
        model_long = self.model_long
        model_mid = self.model_mid
        model_short = self.model_short

        scheduler = self.noise_scheduler

        condition_data_long = condition_data[:, self.num_ctrl_pts:, :]
        condition_data_mid = condition_data[:, self.num_ctrl_pts:self.num_ctrl_pts*2, :]
        condition_data_short = condition_data[:, :self.num_ctrl_pts, :]

        condition_mask_long = condition_mask[:, self.num_ctrl_pts:, :]
        condition_mask_mid = condition_mask[:, self.num_ctrl_pts:self.num_ctrl_pts*2, :]
        condition_mask_short = condition_mask[:, :self.num_ctrl_pts, :]

        trajectory_long = torch.randn(
            size=condition_data_long.shape, 
            dtype=condition_data_long.dtype,
            device=condition_data_long.device,
            generator=generator)

        trajectory_mid = torch.randn(
            size=condition_data_mid.shape, 
            dtype=condition_data_mid.dtype,
            device=condition_data_mid.device,
            generator=generator)

        trajectory_short = torch.randn(
            size=condition_data_short.shape, 
            dtype=condition_data_short.dtype,
            device=condition_data_short.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory_long[condition_mask_long] = condition_data_long[condition_mask_long]
            trajectory_mid[condition_mask_mid] = condition_data_mid[condition_mask_mid]
            trajectory_short[condition_mask_short] = condition_data_short[condition_mask_short]

            # 2. predict model output
            pre_model_output_long = model_long(trajectory_long, t, 
                local_cond=local_cond, global_cond=global_cond)
                        
            pre_model_output_mid = model_long(trajectory_mid, t, 
                local_cond=local_cond, global_cond=global_cond)

            pre_model_output_short = model_long(trajectory_short, t, 
                local_cond=local_cond, global_cond=global_cond)

            batch_size, _, act_dim = pre_model_output_long.shape

            token_long = pre_model_output_long.reshape(batch_size, -1)
            token_mid = pre_model_output_mid.reshape(batch_size, -1)
            token_short = pre_model_output_short.reshape(batch_size, -1)

            token_long  = self.proj_up(token_long)   # (batch_size, hidden_size)    
            token_mid   = self.proj_up(token_mid)
            token_short = self.proj_up(token_short)

            tokens = torch.stack([token_long, token_mid, token_short], dim=1)  # (batch_size, 3, hidden_size)

            inputs = torch.cat([self.cls_tokens_expand, tokens], dim=1)                    # (batch_size, 6, hidden_size)

            output = self.cross_attention(inputs_embeds=inputs)

            cls_outputs = output.last_hidden_state[:, :3, :]  # (batch_size, 3, hidden_size)

            final_pred = self.proj_back(cls_outputs)   # (batch_size, 3, act_dim * num_ctrl_pts)
            final_pred = final_pred.reshape(batch_size, 3, num_ctrl_pts, act_dim)

            model_output_long  = final_pred[:, 0, :, :]  # (batch_size, num_ctrl_pts, act_dim)
            model_output_mid   = final_pred[:, 1, :, :]
            model_output_short = final_pred[:, 2, :, :]

            # 3. compute previous image: x_t -> x_t-1
            trajectory_long = scheduler.step(
                model_output_long, t, trajectory_long, 
                generator=generator,
                **kwargs
                ).prev_sample

            trajectory_mid = scheduler.step(
                model_output_mid, t, trajectory_mid, 
                generator=generator,
                **kwargs
                ).prev_sample

            trajectory_short = scheduler.step(
                model_output_short, t, trajectory_short, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory_long[condition_mask_long] = condition_data_long[condition_mask_long] 
        trajectory_mid[condition_mask_mid] = condition_data_mid[condition_mask_mid] 
        trajectory_short[condition_mask_short] = condition_data_short[condition_mask_short]

        trajectory = torch.cat([target_long, trajectory_mid, trajectory_short], dim=1)        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon  
        Da = self.action_dim  ## 动作维度
        Do = self.obs_feature_dim  ## 观察特征维度
        To = self.n_obs_steps  ## 观察步数

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, self.num_ctrl_pts * 3, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # 后面这种情况基本不用，所以先不改这里的代码
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        # 先假定这个 naction_pred 的形状是(B, self.num_ctrl_pts * 3, act_dim)

        naction_pred = nsample[...,:Da] # 最后一维应该是 act_dim
        ctrl_pts_pred = self.normalizer['action'].unnormalize(naction_pred)

        ctrl_pts_pred_long = ctrl_pts_pred[:, self.num_ctrl_pts:, :]
        ctrl_pts_pred_mid = ctrl_pts_pred[:, self.num_ctrl_pts:self.num_ctrl_pts*2, :]
        ctrl_pts_pred_short = ctrl_pts_pred[:, :self.num_ctrl_pts, :]
        
        # 执行上采样
        action_pred = bezier_upsample(ctrl_pts_pred_long, ctrl_pts_pred_mid, ctrl_pts_pred_short, self.horizon)
        # 形状 (batch_size, horizon / 4 = 16, act_dim)


        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end, :]
        # 预测了 12 步，实际执行了 8 步


        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        device = self.device


        # 生成三个层级控制点的 ground truth 
        nactions = nactions.transpose(1, 2) # (batch_size, act_dim, horizon)
        # (batch_size, act_dim, horizon) -> (batch_size, act_dim, num_ctrl_pts)
        gt_control_pts_long = self.data_fitter_long.fit(nactions)
        gt_control_pts_mid = self.data_fitter_mid.fit(nactions[:, :, :nactions.shape[2]//2])
        gt_control_pts_short = self.data_fitter_short.fit(nactions[:, :, :nactions.shape[2]//4])

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # 试图交换维度顺序 （B, h, t) -> (B, t, h)
        trajectory_long = torch.tensor(gt_control_pts_long, device=device, dtype=torch.float32).transpose(1, 2)  ## 把轨迹改成 gt_control_pts_long ，就是拟合的控制点
        trajectory_mid = torch.tensor(gt_control_pts_mid, device=device, dtype=torch.float32).transpose(1, 2)  ## 把轨迹改成 gt_control_pts_mid ，就是拟合的控制点
        trajectory_short = torch.tensor(gt_control_pts_short, device=device, dtype=torch.float32).transpose(1, 2)  ## 把轨迹改成 gt_control_pts_long ，就是拟合的控制点


        cond_data_long = trajectory_long  
        cond_data_mid = trajectory_mid
        cond_data_short = trajectory_short  

        # 由于默认配置是 obs_as_global_cond == true，所以另外一种情况先不考虑
        # 下面是 obs 编码，与我们的改进无关，所以先不改
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory_long = cond_data.detach()

        # generate impainting mask  # 这里有维度顺序不对引发的问题，需要整体调整  # 目前调好了
        condition_mask_long = self.mask_generator(trajectory_long.shape)
        condition_mask_mid = self.mask_generator(trajectory_mid.shape)
        condition_mask_short = self.mask_generator(trajectory_short.shape)

        # Sample noise_long that we'll add to the images
        ################################################
        # 生成三个层级的噪声
        noise_long = torch.randn(trajectory_long.shape, device=trajectory_long.device)
        noise_mid = torch.randn(trajectory_mid.shape, device=trajectory_mid.device)
        noise_short = torch.randn(trajectory_short.shape, device=trajectory_short.device)

        bsz = trajectory_long.shape[0]

        # Sample a random timestep for each image
        # 使用共享的时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory_long.device
        ).long()
        
        # Add noise_long to the clean images according to the noise_long magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory_long = self.noise_scheduler.add_noise(
            trajectory_long, noise_long, timesteps)
        
        noisy_trajectory_mid = self.noise_scheduler.add_noise(
            trajectory_mid, noise_mid, timesteps)

        noisy_trajectory_short = self.noise_scheduler.add_noise(
            trajectory_short, noise_short, timesteps)
        ################################################

        # compute loss mask
        loss_mask_long = ~condition_mask_long
        loss_mask_mid = ~condition_mask_mid
        loss_mask_short = ~condition_mask_short

        # apply conditioning
        noisy_trajectory_long[condition_mask_long] = cond_data_long[condition_mask_long]
        noisy_trajectory_mid[condition_mask_mid] = cond_data_mid[condition_mask_mid]
        noisy_trajectory_short[condition_mask_short] = cond_data_short[condition_mask_short]


        # Predict the noise_long residual  去噪过程
        pred_long, pred_mid, pred_short = self.model(noisy_trajectory_long, noisy_trajectory_mid, 
            noisy_trajectory_short, timesteps,
            local_cond=local_cond, global_cond=global_cond)  

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target_long = noise_long
            target_mid = noise_mid
            target_short = noise_short
            
        elif pred_type == 'sample':
            target_long = trajectory_long
            target_mid = trajectory_mid
            target_short = trajectory_short
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        ################################################
        loss_long = F.mse_loss(pred_long, target_long, reduction='none')
        loss_long = loss_long * loss_mask_long.type(loss_long.dtype)
        loss_long = reduce(loss_long, 'b ... -> b (...)', 'mean')
        loss_long = loss_long.mean()

        loss_mid = F.mse_loss(pred_mid, target_mid, reduction='none')
        loss_mid = loss_mid * loss_mask_mid.type(loss_mid.dtype)
        loss_mid = reduce(loss_mid, 'b ... -> b (...)', 'mean')
        loss_mid = loss_mid.mean()

        loss_short = F.mse_loss(pred_short, target_short, reduction='none')
        loss_short = loss_short * loss_mask_short.type(loss_short.dtype)
        loss_short = reduce(loss_short, 'b ... -> b (...)', 'mean')
        loss_short = loss_short.mean()

        ## 这个比例还可以调，目前这种分配是希望模型多学习长远的规划
        loss = (loss_long * 1.25 + loss_mid * 1 + loss_short * 0.75) / 3
        ################################################

        return loss
