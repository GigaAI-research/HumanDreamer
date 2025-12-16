from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from ..mdm.mdm_diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal1D, get_down_block, get_up_block
from diffusers.utils import BaseOutput, logging
from ...nn import PatchEmbed1D
from einops import rearrange
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class MotionDiTOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None

class MotionDiT(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            'CrossAttnDownBlockSpatioTemporal1D',
            'CrossAttnDownBlockSpatioTemporal1D',
            'CrossAttnDownBlockSpatioTemporal1D',
            'DownBlockSpatioTemporal',
        ),
        up_block_types: Tuple[str] = (
            'UpBlockSpatioTemporal1D',
            'CrossAttnUpBlockSpatioTemporal1D',
            'CrossAttnUpBlockSpatioTemporal1D',
            'CrossAttnUpBlockSpatioTemporal1D',
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = None,
        loss_type: str = None,
        patch_size: Optional[int] = None,
        need_attn: bool = False,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f'Must provide the same number of `down_block_types` as `up_block_types`. '
                f'`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}.'
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `block_out_channels` as `down_block_types`. '
                f'`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `num_attention_heads` as `down_block_types`. '
                f'`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}.'
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `cross_attention_dim` as `down_block_types`. '
                f'`cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}.'
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `layers_per_block` as `down_block_types`. '
                f'`layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}.'
            )

        # input
        if self.patch_size is None:
            self.conv_in = nn.Conv1d(
                in_channels,
                block_out_channels[0],
                kernel_size=3,
                padding=1,
            )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i] if cross_attention_dim is not None else None,
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn='silu',
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal1D(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1] if cross_attention_dim is not None else None,
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim)) if cross_attention_dim is not None else None
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i] if cross_attention_dim is not None else None,
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn='silu',
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if self.patch_size is None:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
            self.conv_act = nn.SiLU()

            self.conv_out = nn.Conv1d(
                block_out_channels[0],
                out_channels,
                kernel_size=3,
                padding=1,
            )
        self.loss_type = loss_type
        if loss_type == 'fe2':
            # feature projection
            self.fe_proj = FeatureProjection(num_frames)
        self.num_frames = num_frames
        # patch embedding
        self.patch_size = patch_size
        if self.patch_size is not None:
            self.pos_embed = PatchEmbed1D(
                num_points_latent=16,  # TODO num_points_latent = num_points/8
                patch_size=self.patch_size,
                in_channels=4,#in_channels,
                embed_dim=320,#inner_dim,
            )
            self.unpatch_embed = nn.Linear(320, 8)
        # add attn on f*16
        self.need_attn = need_attn
        if need_attn:
            self.full_attn = nn.TransformerEncoderLayer(d_model=1280, nhead=8)
        
    def set_num_frames(self, num_frames):
        self.num_frames = num_frames

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, 'get_processor'):
                processors[f'{name}.processor'] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f'{name}.{sub_name}', child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f'A dict of processors was passed, but the number of processors {len(processor)} does not match the'
                f' number of attention layers: {count}. Please make sure to pass {count} processor classes.'
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, 'set_processor'):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f'{name}.processor'))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f'{name}.{sub_name}', child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """Disables custom attention processors and sets the default attention
        implementation."""
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f'Cannot call `set_default_attn_processor` when attention processors are of type '
                f'{next(iter(self.attn_processors.values()))}'
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f'Make sure to set `dim` to either 0 or 1, not {dim}')

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, 'set_chunk_feed_forward'):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,  # [b*f,c,n]  如果上一步用了video_chunk_size，则应当是[2v,c,n]
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        conv_in_additional_residual: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[MotionDiTOutput, Tuple]:
        # 1. time
        timesteps = timestep  # 长度为b*f的list，每个元素是一个timestep，值都相同
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == 'mps'
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0] // self.num_frames

        timesteps = timesteps.expand(sample.shape[0]) # [b*f]
        t_emb = self.time_proj(timesteps)  # [b*f, 320]

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())  # [bf*3, 256]
        time_embeds = time_embeds.reshape((added_time_ids.shape[0], -1))  # [bf, 768]
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)  # [bf, 1280]
        emb = emb + aug_emb  # [bf, 1280]


        # 2. pre-process
        if self.patch_size is not None:
            sample = self.pos_embed(sample) # [b*f, c, num_points_latent] => [b*f, num_points_patch, embed_dim]=[192,4,16]->[192,8,4]
            sample = rearrange(sample, 'bf n c -> bf c n')  # [b*f, 320, 8]
        else:
            sample = self.conv_in(sample)  # [bf, 320, num_points_latent]

        if conv_in_additional_residual is not None:
            sample = sample + conv_in_additional_residual

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        image_only_indicator = torch.zeros(batch_size, self.num_frames, dtype=sample.dtype, device=sample.device)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )


            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)                

            down_block_res_samples = new_down_block_res_samples
        sample_feature = None
        if self.loss_type == 'fe2':
            sample_feature = self.fe_proj(sample)  # [b*f, 1280, 4]->[b, 256]
        
        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )
        # Self Attention on f*num_points_latent
        # 在f*num_points_latent维度上做self-attention
        if self.need_attn:
            sample = rearrange(sample, '(b f) c n -> (f n) b c', b=batch_size)
            sample = self.full_attn(sample)
            sample = rearrange(sample, '(f n) b c -> (b f) c n', f=self.num_frames, b=batch_size)
        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,  # [b*f,1280,16]
                    temb=emb,  # [b*f,1280]
                    res_hidden_states_tuple=res_samples,  # [b*f,320,16][b*f,640,16][b*f,640,16]
                    encoder_hidden_states=encoder_hidden_states,  # [b*f,1,1024]
                    image_only_indicator=image_only_indicator,  # [b,f]
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        # unpatchify
        if self.patch_size is not None:
            # unpatch方式：用nn.linear在特征维度上乘2，然后reshape到num_points_patch维度上，这样能够恢复为num_points_latent
            sample = rearrange(sample, 'bf c n -> bf n c')
            
            sample = self.unpatch_embed(sample) # [bf,num_points_patch,c]->[bf,num_points_patch,2c]
            num_points_patch = sample.shape[1]
            sample = sample.reshape(shape=(-1, num_points_patch, self.patch_size, 4))  # [b*f, num_points_patch, patch_size, c]
            sample = torch.einsum('nlpc->nclp', sample)  # [b*f, c, num_points_patch, patch_size]
            sample = sample.reshape(shape=(-1, 4, num_points_patch*self.patch_size))
        else:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)  # [b*f, c, num_points_patch]

        if not return_dict:
            return (sample,sample_feature)

        return MotionDiTOutput(sample=sample,sample_feature=sample_feature)

class FeatureProjection(nn.Module):
    def __init__(self, num_frames):
        super(FeatureProjection, self).__init__()
        self.num_frames = num_frames
        self.mlp = MLP(dim=1280, projection_size=256)

    def forward(self, sample):
        b_f, c, w = sample.shape
        b = b_f // self.num_frames
        f = b_f // b
        sample = rearrange(sample, 'bf c n -> (bf n) c')
        
        sample = self.mlp(sample) # [bf*8, 1280] -> [bf*8, 256]
        sample = rearrange(sample, '(bf n) c -> bf n c', n=w)
        sample = sample.view(b, f, w, -1)
        sample = sample.mean(dim=1)
        x = sample.mean(dim=1)  # 或者可以使用最大池化 sample.max(dim=1).values
        return x  # [b,256]


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=2):
        super().__init__()
        self.in_features = dim
        assert num_layer==2
        if num_layer == 1:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == 2:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)