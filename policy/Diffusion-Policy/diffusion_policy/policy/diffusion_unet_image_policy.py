from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.bezier_curve.data_fit_bezier_curve import BezerFitter

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

        model = ConditionalUnet1D(
            input_dim=input_dim,
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

        ###################################################################################
        ## 贝塞尔控制点拟合器
        self.num_inference_steps = num_inference_steps
        self.data_fitter_long = BezerFitter(input_dim=14, num_control_points=5, horizon_length=64)
        self.data_fitter_mid = BezerFitter(input_dim=14, num_control_points=5, horizon_length=32)
        self.data_fitter_short = BezerFitter(input_dim=14, num_control_points=5, horizon_length=16)
        ###################################################################################
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask_long,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask_long] = condition_data[condition_mask_long]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask_long] = condition_data[condition_mask_long]        

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
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
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
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
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

        ################################################
        ## 生成三个层级控制点的 ground truth 
        gt_control_pts_long = data_fitter_long.fit(nactions)
        gt_control_pts_mid = data_fitter_mid.fit(nactions[:, :nactions.shape[1]//2])
        gt_control_pts_short = data_fitter_short.fit(nactions[:, :nactions.shape[1]//4])
        ################################################

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        ################################################
        trajectory_long = gt_control_pts_long  ## 把轨迹改成 gt_control_pts_long ，就是拟合的控制点
        trajectory_mid = gt_control_pts_mid  ## 把轨迹改成 gt_control_pts_mid ，就是拟合的控制点
        trajectory_short = gt_control_pts_short  ## 把轨迹改成 gt_control_pts_long ，就是拟合的控制点
        ################################################


        cond_data_long = trajectory_long  
        cond_data_mid = trajectory_mid
        cond_data_short = trajectory_short  

        ## 由于默认配置是 obs_as_global_cond == true，所以另外一种情况先不考虑
        ## 下面是 obs 编码，与我们的改进无关，所以先不改
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

        # generate impainting mask
        condition_mask_long = self.mask_generator(trajectory_long.shape)
        condition_mask_mid = self.mask_generator(trajectory_mid.shape)
        condition_mask_short = self.mask_generator(trajectory_short.shape)

        # Sample noise_long that we'll add to the images
        ################################################
        # 生成三个层级的噪声
        noise_long = torch.randn(trajectory_long.shape, device=trajectory_long.device)
        noise_mid = torch.randn(trajectory_mid.shape, device=trajectory_mid.device)
        noise_short = torch.randn(trajectory_short.shape, device=trajectory_short.device)

        bsz_long = trajectory_long.shape[0]
        bsz_mid = trajectory_mid.shape[0]
        bsz_short = trajectory_short.shape[0]

        # Sample a random timestep for each image
        timesteps_long = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz_long,), device=trajectory_long.device
        ).long()
        
        timesteps_mid = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz_mid,), device=trajectory_long.device
        ).long()
        
        timesteps_short = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz_short,), device=trajectory_long.device
        ).long()

        # Add noise_long to the clean images according to the noise_long magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory_long = self.noise_scheduler.add_noise(
            trajectory_long, noise_long, timesteps_long)
        
        noisy_trajectory_mid = self.noise_scheduler.add_noise(
            trajectory_mid, noise_mid, timesteps_mid)

        noisy_trajectory_short = self.noise_scheduler.add_noise(
            trajectory_short, noise_short, timesteps_short)
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
        ## 设法让去噪过程返回一个元组，model 还要修改
        pred_long, pred_mid, pred_short = self.model(noisy_trajectory_long, timesteps_long, 
            noisy_trajectory_mid, timesteps_mid, noisy_trajectory_short, timesteps_short,
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
