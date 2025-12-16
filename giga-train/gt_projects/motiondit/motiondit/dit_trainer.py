import functools
import math
import torch
from einops import rearrange
from diffusers.optimization import get_scheduler
from einops import rearrange
from giga_datasets import DefaultCollator, DefaultSampler, TrainTestSampler,load_dataset, CLIPTextTransform
from giga_models.nn import ModuleDict
from giga_models import AutoencoderKLTemporalDecoder1d, MotionDiT, TMR
from giga_train import Trainer
from .dit_transforms import DITTransform

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class DITTrainer(Trainer):
    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        batch_size = data_config.batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform_cfg = data_config.transform
        transform_type = transform_cfg.pop('type')
        num_frames = data_config.get('num_frames',None)
        self.mask = torch.ones((1,num_frames,1,1),device=self.device)

        transform = DITTransform(**transform_cfg)
        dataset.set_transform(transform)
        sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
        sampler_type = sampler_cfg.pop('type')
        if sampler_type == 'DefaultSampler':
            sampler = DefaultSampler(dataset, batch_size=batch_size, **sampler_cfg)
        elif sampler_type == 'TrainTestSampler':
            sampler = TrainTestSampler(dataset, batch_size=batch_size, **sampler_cfg)
        else:
            assert False
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=DefaultCollator(),
            batch_size=data_config.batch_size_per_gpu,
            num_workers=data_config.num_workers,
            pin_memory=True,
        )
        self.ref_frames = data_config.get('ref_frames', 0)
        return dataloader

    def get_models(self, model_config):
        self.train_mode = model_config.get('train_mode', None)
        self.mask_type = model_config.get('mask_type', None)
        if self.train_mode == 'vae':
            vae_cfg = model_config.get('vae', {})
            vae_dtype = vae_cfg.get('dtype', 'float32')
            vae_dtype = getattr(torch, vae_dtype)
            vae_hp = vae_cfg.get('hyperparameters', {})
            vae = AutoencoderKLTemporalDecoder1d(**vae_hp)
            vae.requires_grad_(True)
            vae.to(self.device, dtype=vae_dtype)
            models = [vae]
            checkpoint = model_config.get('checkpoint', None)
            self.load_checkpoint(checkpoint, models)
            
            # load pretrain ckpt
            load_vae_ckpt = False
            if load_vae_ckpt:
                vae_weight_path = model_config.get('vae_weight_path', None)
                print('loading vae checkpoint from {}'.format(vae_weight_path))
                vae_ckpt = torch.load(vae_weight_path, map_location='cpu')
                vae.load_state_dict(vae_ckpt, strict=True)

            model = dict(vae=vae)
            model = ModuleDict(model)
            model.train()

            # 下面代码是减少显存占用
            if model_config.get('activation_checkpointing', False):
                cls_names = 'BasicTransformerBlock,TemporalBasicTransformerBlock,SpatioTemporalResBlock'
                cls_names = cls_names.split(',')
                cls_to_wrap = set()
                for cls_name in cls_names:
                    transformer_cls = get_module_class_from_name(model, cls_name)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        cls_to_wrap.add(transformer_cls)
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=cls_to_wrap,
                )
                apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)
            model.to(self.device)
            
        elif self.train_mode == 'dit':
            # VAE
            vae_cfg = model_config.get('vae', {})
            vae_dtype = vae_cfg.get('dtype', 'float32')
            vae_dtype = getattr(torch, vae_dtype)
            vae_hp = vae_cfg.get('hyperparameters', {})
            self.vae = AutoencoderKLTemporalDecoder1d(**vae_hp)
            self.vae.requires_grad_(False)
            self.vae.to(self.device, dtype=vae_dtype)
            vae_weight_path = model_config.get('vae_weight_path', None)  # 这是测试VAE效果用的
            print('loading vae checkpoint from {}'.format(vae_weight_path))
            vae_ckpt = torch.load(vae_weight_path, map_location='cpu')
            self.vae.load_state_dict(vae_ckpt, strict=True)

            # MotionDiT
            dit_cfg = model_config.get('dit',{})
            dit_hp = dit_cfg.get('hyperparameters', {})
            dit = MotionDiT(**dit_hp)
            load_dit_ckpt = False
            if load_dit_ckpt:
                dit_weight_path = model_config.get('dit_weight_path', None)
                self.logger.info(f'loading dit checkpoint from {dit_weight_path}')
                dit_ckpt = torch.load(dit_weight_path, map_location='cpu')
                dit.load_state_dict(dit_ckpt, strict=False)
            self.gen_type = model_config.get('gen_type', 'it2m')
            if self.activation_checkpointing:
                dit.enable_gradient_checkpointing()  # 功能：激活梯度检查点，减少显存占用，但会增加计算时间
            model = dict(dit=dit)
            model = ModuleDict(model)
            model.train()
            
            # cal params
            pytorch_total_params = sum(p.numel() for p in dit.parameters())
            trainable_pytorch_total_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
            print('Total - ', pytorch_total_params)
            print('Trainable - ', trainable_pytorch_total_params)

            # 下面代码是减少显存占用
            if model_config.get('activation_checkpointing', False):
                cls_names = 'BasicTransformerBlock,TemporalBasicTransformerBlock,SpatioTemporalResBlock'
                cls_names = cls_names.split(',')
                cls_to_wrap = set()
                for cls_name in cls_names:
                    transformer_cls = get_module_class_from_name(model, cls_name)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        cls_to_wrap.add(transformer_cls)
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=cls_to_wrap,
                )
                apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)
            model.to(self.device)

        clip_path ='Path to stabilityai/stable-diffusion-2-1-base on huggingface'
        self.text_transform = CLIPTextTransform(model_path=clip_path, device=self.device)
        self.loss_type = model_config.get('loss_type', 'normal')
        if self.loss_type == 'feature':
            tmr_weight_path = 'Your_exp_root_path/clop/clop_f64_ALL_wholebody_vae1d_kl_s2_b32/models/checkpoint_epoch_5_step_21010/clop/diffusion_pytorch_model.bin'
            tmr_config = dict(
                dtype='float32',
                tmr_weight_path = tmr_weight_path,
                hyperparameters=dict(
                    motion_encoder_cfg=dict(
                                        nfeats=256,
                                        vae= True,
                                        latent_dim= 256,
                                        ff_size= 1024,
                                        num_layers= 6,
                                        num_heads= 4,
                                        dropout= 0.1,
                                        activation= 'gelu'),
                    text_encoder_cfg=dict(
                                        nfeats= 1024,
                                        vae= True,
                                        latent_dim= 256,
                                        ff_size= 1024,
                                        num_layers= 6,
                                        num_heads= 4,
                                        dropout= 0.1,
                                        activation= 'gelu'),
                    motion_decoder_cfg=dict(
                                        nfeats= 256,
                                        latent_dim= 256,
                                        ff_size= 1024,
                                        num_layers= 6,
                                        num_heads= 4,
                                        dropout= 0.1,
                                        activation= 'gelu'),
                    lr = 1e-4,
                    vae=True,
                    fact=None, # Optional[float] = None,
                    sample_mean=False, # Optional[bool] = False,
                    lmd=dict({"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1}),
                    temperature = 0.7,
                    threshold_selfsim = 0.80,
                    threshold_selfsim_metrics = 0.95,
                    )
                )
            tmr_dtype = tmr_config.get('dtype', 'float32')
            tmr_dtype = getattr(torch, tmr_dtype)
            tmr_hp = tmr_config.get('hyperparameters', {})
            tmr_weight_path = tmr_config.get('tmr_weight_path', None)
            self.clop = TMR(**tmr_hp)
            self.clop.requires_grad_(False)
            self.clop.to(self.device, dtype=tmr_dtype)
            if tmr_weight_path is not None:
                self.clop.load_state_dict(torch.load(tmr_weight_path, map_location=self.device))
        return model

    def get_schedulers(self, scheduler):
        scheduler = get_scheduler(
            name=scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=scheduler.num_warmup_steps * self.num_processes,
            num_training_steps=self.max_steps * self.num_processes,
        )
        return scheduler

    def forward_step(self, batch_dict):
        # Prepare input
        subset_tensor = batch_dict['subset_array'].to(torch.float32).contiguous()  # [b,f,n,1]
        rela_pose_tensor = batch_dict['pose_array'].to(torch.float32).contiguous()  # [b,f,num_points,c]
        batch_size, num_frames, num_points = rela_pose_tensor.shape[:3]
        if self.mask_type=='score':
            score_tensor = batch_dict['score_array'].to(torch.float32)
            input_pose = torch.cat([rela_pose_tensor, score_tensor], dim=-1)
        else:
            input_pose = rela_pose_tensor  # [b,f,num_points,2]

        if self.train_mode == 'vae':
            vae = functools.partial(self.model, 'vae')
                
            self.set_num_frames(num_frames, self.model)
            
            input_pose = rearrange(input_pose,'b f n c -> (b f) c n')  # [b*f,c,num_points]
            pred_motion, posterior = vae(sample=input_pose,
                            sample_posterior=True,  # 训练时使用采样，增强随机性
                            num_frames=num_frames,
                            return_dict=False,
                            return_posterior=True)
            
            input_pose = rearrange(input_pose,'(b f) c n -> b f n c', b=batch_size)  # [b,f,num_points,2]
            pred_motion = rearrange(pred_motion,'(b f) c n -> b f n c', b=batch_size)  # [b,f,num_points,2]
            kl_loss = posterior.kl()  
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # 前2个通道重复，第3个通道全是1
            if self.mask_type=='score':
                subset_tensor = subset_tensor.repeat(1,1,1,3)
                subset_tensor[:,:,:,2] = 1  # 不能给第三个通道乘，因为第三个通道是mask，乘了之后学不到了
            
            input_pose = input_pose * subset_tensor
            pred_motion = pred_motion * subset_tensor
            pose_recon_loss = torch.mean((input_pose - pred_motion) ** 2)
            total_loss = pose_recon_loss + 1e-7 * kl_loss

            return total_loss
        elif self.train_mode == 'dit':
            # prepare model
            dit = functools.partial(self.model, 'dit')
            self.set_num_frames(num_frames, self.model)
            
            ref_frames = self.ref_frames
            if ref_frames!=1 and self.gen_type=='it2m':
                raise NotImplementedError
            ref_pose = input_pose[:,:ref_frames]  # [b,ref_f,num_points,2]

            input_pose = rearrange(input_pose,'b f n c -> (b f) c n')  # [b*f,c,num_points]
            ref_pose = rearrange(ref_pose,'b f n c -> (b f) c n')  # [b*ref_f,c,num_points]
            with torch.no_grad():
                latents = self.vae.encode(input_pose).latent_dist.sample()  # [b*f,latent_c,num_points_latent]
                latents = latents * self.vae.config.scaling_factor  # [b*f,latent_c,num_points_latent]
                latents = rearrange(latents, '(b f) c n -> b f c n', f=num_frames)  # [b,f,latent_c,num_points_latent]

            if self.gen_type == 'it2m':
                # conditional_latents
                with torch.no_grad():
                    conditional_latents = self.vae.encode(ref_pose).latent_dist.sample()  # [b*ref_f,latent_c,num_points_latent]
                    
                drop_prob = 0.2
                if drop_prob > 0:
                    random_p = torch.rand(batch_size*ref_frames, device=self.device)[:, None, None]
                    image_mask = 1 - ((random_p >= drop_prob) * (random_p < 3 * drop_prob)).float()
                    conditional_latents = conditional_latents * image_mask  # [b*ref_f,4,num_points_latent]
                conditional_latents = rearrange(conditional_latents,'(b f0) c n -> b f0 c n', f0=ref_frames)  # [b,ref_f,4,num_points_latent]
                # new_conditional_latents:前ref_frames帧是conditional_latents，后面的帧是conditional_latents最后一帧的重复
                new_conditional_latents = torch.zeros_like(latents, device=self.device)  # [b,f,4,num_points_latent]
                new_conditional_latents[:, 0:ref_frames] = conditional_latents
                new_conditional_latents[:, ref_frames:] = conditional_latents[:, -1:].repeat_interleave(num_frames - ref_frames, dim=1)
                new_conditional_latents = rearrange(new_conditional_latents, 'b f c n -> (b f) c n')  # [b*f,4,num_points_latent]
            
            # add noise
            noise = torch.randn_like(latents)  # [b*f,4,num_points/8]
            sigmas = rand_cosine_interpolated(shape=(batch_size,)).to(self.device)  # [b]=[1]
            sigmas = sigmas[:, None, None, None]  # [b,1,1,1]=[1,1,1,1]
            
            noisy_latents = latents + noise * sigmas  # [b,f,4,num_points/8]  /  [b,f,c,n]
            timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(self.device)
            inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)  # [b, f, 4, num_points/8]

            timesteps = timesteps.repeat_interleave(num_frames, dim=0)  # [b*f]=[16]
            # added_time_ids
            fps = 6
            motion_bucket_id = 127
            noise_aug_strength = 0.02
            added_time_ids = [fps, motion_bucket_id, noise_aug_strength]
            added_time_ids = torch.tensor([added_time_ids], device=self.device)
            added_time_ids = torch.cat([added_time_ids] * batch_size * num_frames)
            # Predict the noise residual
            with self.accelerator.autocast():
                inp_noisy_latents = inp_noisy_latents.flatten(0, 1)  # [b*f, 4, h/8, w/8]
                if self.gen_type == 'it2m':
                    inp_noisy_latents = torch.cat([inp_noisy_latents, new_conditional_latents], dim=1)  # [b*f, 8,num_points_latent]
                prompt_embeds = self.text_transform(batch_dict['prompt'],mode='after_pool',to_numpy=False)  # [b,1024]
                prompt_embeds = rearrange(prompt_embeds, 'b c -> b 1 c')
                prompt_embeds = prompt_embeds.repeat_interleave(num_frames, dim=0)  # [b*f, 1, 1024]=[16,1,1024]

                model_pred, sample_feature = dit(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_time_ids=added_time_ids,
                    return_dict = False,
                    )
                model_pred = rearrange(model_pred, '(b f) c n -> b f c n', f=num_frames)

            # Denoise the latents
            c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
            c_skip = 1 / (sigmas**2 + 1)
            weighing = (1 + sigmas**2) * (sigmas**-2.0)
            denoised_latents = (model_pred * c_out + c_skip * noisy_latents) * self.mask
            target = latents * self.mask
            
            loss = torch.mean(
                (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                dim=1,
            )

            if self.loss_type == 'feature':
                # dit中某一层feature通过MLP后与GT pose的CLOP feature对齐
                x_dict={}
                input_pose = rearrange(input_pose,'(b f) c n -> b f n c', f=num_frames)
                x_dict['x'] = rearrange(input_pose[:,:,:,:2], 'b f n c -> b f (n c)')  # [b,f,pose_dim]
                x_dict['mask'] = torch.full((batch_size, self.num_frames), True, dtype=torch.bool).to(self.device)
                with torch.inference_mode():
                    gt_feature = self.clop.encode(x_dict, sample_mean=True)  # [b,256]
                feature2_loss = torch.mean((sample_feature - gt_feature) ** 2)
                loss = loss + feature2_loss * 1e-3
            return loss


    def set_num_frames(self, num_frames, model):
        if not hasattr(self, 'num_frames'):
            self.num_frames = -1
        if self.num_frames != num_frames:
            if hasattr(model, 'module'):
                model = model.module
            for name, module in model.named_modules():
                if hasattr(module, 'set_num_frames'):
                    module.set_num_frames(num_frames)
            self.num_frames = num_frames


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f'groups must be positive, got {groups}')
    if group < 0 or group >= groups:
        raise ValueError(f'group must be in [0, {groups})')
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n

def rand_cosine_interpolated(
    shape,
    image_d=64,
    noise_d_low=32,
    noise_d_high=64,
    sigma_data=0.5,
    min_value=0.002,
    max_value=700,
    device='cpu',
    dtype=torch.float32,
):
    """Draws samples from an interpolated cosine timestep distribution (from
    simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
