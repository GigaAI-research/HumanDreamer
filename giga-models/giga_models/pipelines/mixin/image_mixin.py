import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ... import utils
from .utils import add_control_model_name


class ImageMixin:
    def load_clip(self, pretrained_model_path):
        if self.feature_extractor is None:
            self.feature_extractor = CLIPImageProcessor()
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_path,
            torch_dtype=self.dtype,
            local_files_only=utils.is_offline_mode(),
        )
        self.clip_image_encoder.to(self.device)
        add_control_model_name('clip_image_encoder')

    def get_timesteps(self, num_inference_steps, strength, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1
            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps
        return timesteps, num_inference_steps - t_start

    def repeat_data(
        self,
        data,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        do_classifier_free_guidance=False,
    ):
        if num_frames is not None:
            total_size = batch_size * num_images_per_prompt * num_frames
        else:
            total_size = batch_size * num_images_per_prompt
        if data.shape[0] == 1:
            repeat_by = total_size
        else:
            repeat_by = num_images_per_prompt
        if repeat_by != 1:
            data = data.repeat_interleave(repeat_by, dim=0)
        if do_classifier_free_guidance:
            data = torch.cat([data] * 2)
        return data

    def encode_vae_image(
        self,
        image,
        height,
        width,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        do_classifier_free_guidance=False,
        sample_mode='sample',
        add_noise=True,
        timestep=None,
        generator=None,
    ):
        if image is None:
            return None
        device = self._execution_device
        dtype = self.dtype
        image = self.image_processor.preprocess(image, height, width)
        image = image.to(device=device, dtype=dtype)
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            image = image.float()
            self.vae.to(dtype=torch.float32)
        latent_dist = self.vae.encode(image).latent_dist
        if sample_mode == 'sample':
            latents = latent_dist.sample(generator)
        elif sample_mode == 'argmax':
            latents = latent_dist.mode()
        else:
            assert False
        if needs_upcasting:
            self.vae.to(dtype)
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype)
        latents = self.repeat_data(
            latents,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        if add_noise:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep)
        return latents

    def encode_clip_image(
        self,
        image,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        do_classifier_free_guidance=False,
    ):
        if image is None:
            return None
        device = self._execution_device
        pixel_values = self.feature_extractor(
            images=image,
            size=dict(height=224, width=224),
            do_center_crop=False,
            return_tensors='pt',
        ).pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.dtype)
        image_embeddings = self.clip_image_encoder(pixel_values).image_embeds.unsqueeze(1)
        image_embeddings = self.repeat_data(
            image_embeddings,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
        )
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        return image_embeddings
