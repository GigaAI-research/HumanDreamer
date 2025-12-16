from typing import Callable, Dict, List, Optional, Union
import inspect
import PIL.Image
from diffusers.schedulers import EulerDiscreteScheduler
import torch
from accelerate import cpu_offload_with_hook
# from diffusers import StableVideoDiffusionPipeline
# from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipelineOutput,
    _append_dims,
    tensor2vid,
)
from diffusers.utils.torch_utils import randn_tensor,is_compiled_module
from einops import rearrange
import copy
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..mixin import (
    ImageMixin,
    PromptMixin,
    DiTMixin,
    get_control_model_names,
)
import torch.nn.functional as F
from ...models import AutoencoderKLTemporalDecoder1d, MotionDiT
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
# 针对pose坐标，一维。对应使用一维dit和一维vae
class StableVideoDiffusionPOSE_IT2VPipeline1D(
    # StableVideoDiffusionPipeline,
    DiffusionPipeline,
    ImageMixin,
    PromptMixin,
    DiTMixin,
):
    '''
    用于pose首帧坐标+text-->pose序列,原理和IT2V相同,但是不用对img做各种处理。
    img通常是首帧,通过vae,变成conditional_latents,和latents进行concat,送入dit
    text直接输入的时候就是prompt_embeds,作为encoder_hidden_states,送入dit
    '''
    
    def __init__(self, 
                 vae: AutoencoderKLTemporalDecoder1d, 
                 image_encoder: CLIPVisionModelWithProjection, 
                 dit: MotionDiT, 
                 scheduler: EulerDiscreteScheduler, 
                 feature_extractor: CLIPImageProcessor):
        # super().__init__(vae, image_encoder, dit, scheduler, feature_extractor)
        # super(StableVideoDiffusionPOSE_IT2VPipeline1D, self).__init__(vae, image_encoder, dit, scheduler, feature_extractor)
        super().__init__()
        
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            dit=dit,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.cn_scheduler = copy.deepcopy(scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def to(self, device=None, dtype=None):
        if device is None and dtype is None:
            return self
        control_model_names = get_control_model_names()
        for model_name in control_model_names:
            model = getattr(self, model_name, None)
            if model is not None:
                model.to(device, dtype)
        super().to(device, dtype)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = 'cuda'):
        super().enable_model_cpu_offload(gpu_id, device)
        device = self._all_hooks[0].hook.execution_device
        control_model_names = get_control_model_names()
        for model_name in control_model_names:
            model = getattr(self, model_name, None)
            if model is not None:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)
    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        num_points,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            num_points // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    def check_inputs(self, image, num_points):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if num_points % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {num_points}.")
    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1)

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])
            # image_latents = torch.cat([image_latents]*2)

        # duplicate image_latents for each generation per prompt, using mps friendly method
        # image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1)

        return image_latents
    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.dit.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.dit.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [b, f, c, num_points] -> [b*f, c, num_points]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, num_points] -> [batch, channels, frames, num_points]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3,)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames
    
    
    @torch.no_grad()
    def __call__(
        self,
        ref_pose: Union[torch.FloatTensor],
        # prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_points: int = 128,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'tensor',
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ['latents'],
        return_dict: bool = True,
        # IT2M/T2M
        prompt_embeds = None,
        gen_type = 'it2m',
        # others
        video_chunk_size: Optional[int] = None,
        fix_cond_frame = False,  # 固定首帧，让latents首帧直接用condition的
    ):
        # 0. Default height and width to dit
        # height = height or self.dit.config.sample_size * self.vae_scale_factor
        # width = width or self.dit.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.dit.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        video_chunk_size = video_chunk_size if video_chunk_size is not None else num_frames
        decode_chunk_size = min(decode_chunk_size, num_frames)
        video_chunk_size = min(video_chunk_size, num_frames)
        # assert num_videos_per_prompt == 1
        
        image = ref_pose  # [b,c,num_points]
        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(image, num_points)

        # 2. Define call parameters
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale
        # batch_size = 1
        batch_size = image.shape[0]
        device = self._execution_device
        total_size = 2 * num_frames if self.do_classifier_free_guidance else num_frames

        # 3. Encode input image
        if prompt_embeds is not None:
            image_embeddings = prompt_embeds  # [1,1,1024]  # 懒得改名字，这里的image_embeddings是通过clip得到的
            # image_embeddings = self.encode_prompt(
            #     prompt=prompt,
            #     negative_prompt=negative_prompt,
            #     batch_size=batch_size,
            #     num_frames=num_frames,
            # )
            if self.do_classifier_free_guidance:
                negative_prompt_embeddings = torch.zeros_like(prompt_embeds)
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                image_embeddings = torch.cat([negative_prompt_embeddings, image_embeddings])  # [2b,1,1024]
        else:
            assert False
            # image_embeddings = self._encode_image(
            #     image, device, num_videos_per_prompt, self.do_classifier_free_guidance
            # )  # [2b,1,1024]
        if image_embeddings.shape[0] != total_size:
            image_embeddings = image_embeddings.repeat_interleave(num_frames, dim=0)  # [2bf,1,1024]
        image_embeddings = image_embeddings.repeat(num_videos_per_prompt, 1, 1)  # [2bf,1,1024]
        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188  # noqa E501
        fps = fps - 1

        # 4. Encode input image using VAE
        # image = self.image_processor.preprocess(image, height=height, width=width)  # device is cpu
        # image = image.unsqueeze(0) # [1,3,32,32]  # 这里不是image不需要做preprocess之类的，和上一行起的作用是一样的
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)  # [b,c,h,w]=[1,3,32,32]
        image = image + noise_aug_strength * noise  # [1,3,32,32]

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image, self.vae.device, num_videos_per_prompt, self.do_classifier_free_guidance
        )  # [2b,4,num_points/8]
        image_latents = image_latents.to(device, dtype=image_embeddings.dtype)
        if image_latents.shape[0] != total_size:
            image_latents = image_latents.repeat_interleave(num_frames, dim=0)  # [2bf,4,h/8,w/8]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)
        added_time_ids = added_time_ids.repeat_interleave(num_frames, dim=0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.cn_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.dit.config.in_channels if gen_type=='it2m' else self.dit.config.in_channels*2
        # 由于prepare_latents中默认是conditionnal的，将channels除以了2.所以上面无conditional时，需要乘以2
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            num_points,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )  # [b,f,4,num_points/8]
        latents = latents.flatten(0, 1)  # [bf,4,num_points/8]


        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim + 1)
        guidance_scale = guidance_scale.flatten(0, 1)  # [bf,1,1,1]

        self._guidance_scale = guidance_scale

        self.set_num_frames(video_chunk_size)

        # prepare input        # 应当是posedreamer的inference模式
        # 参考giga_dd2/giga-models/giga_models/pipelines/gligen/pipeline_stable_video_diffusion_glvgen.py

        # cn_prompt_embeds = batch_dict['prompt_embeds'].to(self.dtype)[:batch_size]  # (1,1,1024)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents  # [2bf,4,h/8,w/8]
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if fix_cond_frame:
                    latent_model_input[0] = image_latents[0] * self.vae.scaling_factor  # 固定首帧，让latents首帧直接用condition的
                    latent_model_input[num_frames] = image_latents[num_frames]*self.vae.scaling_factor
                
                # Concatenate image_latents over channels dimention
                if gen_type == 'it2m':  # 使用首帧，不仅用text控制
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)  # [2bf,8,h/8,w/8]
                
                noise_pred = self.forward_dit(
                    latent_model_input=latent_model_input,  # [2bf,8,num_points_latent]
                    timestep=t,
                    encoder_hidden_states=image_embeddings,# [2bf,1,1024].to(torch.float16).repeat_interleave(num_frames, dim=0), # [2bf,1,1024]
                    added_time_ids=added_time_ids,
                    return_dict=False,
                    cur_step=i,
                    num_frames=num_frames,
                    chunk_size=video_chunk_size,
                )  # [2bf,4,num_points_latent]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop('latents', latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == 'tensor':
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            latents = rearrange(latents, '(b f) c n -> b f c n', f=num_frames)
            latents = latents.to(self.vae.device)
            if fix_cond_frame:
                latents[:,0] = image_latents[num_frames]*self.vae.scaling_factor
            pred_pose_emb = self.decode_latents(latents, num_frames, decode_chunk_size)  # [b,c,f,h,w]
        else:
            assert False

        self.maybe_free_model_hooks()
        return_dict = False
        if not return_dict:
            return pred_pose_emb
        return StableVideoDiffusionPipelineOutput(frames=pred_pose_emb)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps