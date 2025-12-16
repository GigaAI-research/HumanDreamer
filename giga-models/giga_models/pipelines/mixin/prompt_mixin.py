import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ... import utils
from .utils import add_control_model_name


class PromptMixin:
    def load_text_encoder(self, pretrained_model_path):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path,
            local_files_only=utils.is_offline_mode(),
        )
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_path,
            torch_dtype=self.dtype,
            local_files_only=utils.is_offline_mode(),
        )
        self.text_encoder.to(self.device)
        add_control_model_name('text_encoder')

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
    ):
        device = self._execution_device
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))
        prompt_embeds = prompt_embeds.text_embeds.unsqueeze(1)
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.dtype)
        prompt_embeds = self.repeat_data(
            prompt_embeds,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
        )
        # get unconditional embeddings for classifier free guidance
        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !='
                    f' {type(prompt)}.'
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.'
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))
            negative_prompt_embeds = negative_prompt_embeds.text_embeds.unsqueeze(1)
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=self.dtype)
            negative_prompt_embeds = self.repeat_data(
                negative_prompt_embeds,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds
