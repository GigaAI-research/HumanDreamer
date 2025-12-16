# Origin: mdm_f64_dit_patchattn_t2m_score_s2_5w_fe2loss
from .dataset_path_list import motionvid
import os
# 需要首帧约束，t2m=False,带上脸和手的pose信息
num_frames = 64
num_joints = 18  # body
num_points = 128  # body+hands+face
train_mode = 'dit' # vae
gen_type = 't2m' # 'it2m', 'i2m', 't2m'
ref_frames = None # 首帧约束
mask_type = 'score'  # 'binary', 'score', None
loss_type = 'feature'
stride = 2
exp_root_path = 'XXX/HumanDreamer/exp'
vae_checkpoint_name = 'checkpoint_epoch_400_step_193200'
dit_checkpoint_name = 'checkpoint_epoch_100_step_9900'
config_name = os.path.basename(__file__).split('.')[0]

project_dir = f'{exp_root_path}/{train_mode}/{config_name}/'
vae_save_dir = f'{exp_root_path}/vae/{config_name}/vae_vis/{vae_checkpoint_name}'
dit_save_dir = f'{exp_root_path}/dit/{config_name}/dit_vis/{dit_checkpoint_name}'

vae_weight_path = f'{exp_root_path}/vae/f64_vae1d/models/{vae_checkpoint_name}/vae/diffusion_pytorch_model.bin'
dit_weight_path = f'{exp_root_path}/dit/{config_name}/models/{dit_checkpoint_name}/dit'

config = dict(
    resume=True,
    project_dir=project_dir,
    launch=dict(
        # gpu_ids = [0,1,2,3,4,5,6,7],
        gpu_ids = [3],
        # distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            offload_optimizer_device='none',
            offload_param_device='none',
            zero_stage=2,  # 0, 1, 2
        ),
    ),
    dataloaders=dict(
        train=dict(
            num_frames=num_frames,
            data_or_config = [*motionvid],
            batch_size_per_gpu=64,
            num_workers=64,
            sampler=dict(
                type='TrainTestSampler',
                stage = 'train',
                percentage=5/126, # total:1.26M, so 5/126=50k
            ),
            transform=dict(
                type='DITTransform',
                num_frames=num_frames,
                is_train=True,
                num_joints=num_joints,
                train_mode=train_mode,
                mask_type=mask_type,
                stride=stride
            ),
            ref_frames=ref_frames,
        ),
        test=dict(
            data_or_config = [*motionvid[0:9]],
            frame_num=num_frames,
            batch_size_per_gpu=1,
            num_workers=1,
            sampler=dict(
                type='TrainTestSampler',
                stage = 'test',
                percentage=1/1000,
            ),
            transform=dict(
                type='DITTransform',
                num_frames=num_frames,
                is_train=False,
                num_joints=num_joints,
                train_mode=train_mode,
                mask_type=mask_type,
                stride=stride
            ),
            ref_frames=ref_frames,
        ),
    ),
    models=dict(
        pose_dim=num_points*2,
        gen_type=gen_type,
        use_pose_attention=False,
        train_mode=train_mode,
        mask_type=mask_type,
        loss_type = loss_type,
        vae = dict(
            pretrained=None,
            dtype='float32',
            hyperparameters=dict(
                in_channels=3, # x和y,如果有mask则为3
                out_channels=3,
                down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
                block_out_channels=[128,256,512,512],
                layers_per_block=2,
                latent_channels=4,
                sample_size=768,
            )
        ),
        dit=dict(
            dytpe='float32',
            hyperparameters=dict(
                block_out_channels=(320, 640, 1280),
                addition_time_embed_dim=256,
                num_attention_heads=(5, 10, 20),
                projection_class_embeddings_input_dim=768,
                transformer_layers_per_block=1,
                layers_per_block=2,
                sample_size=96,
                in_channels = 4 if gen_type == 't2m' else 8,
                out_channels = 4,
                down_block_types=("CrossAttnDownBlockSpatioTemporal1D_nosample", "CrossAttnDownBlockSpatioTemporal1D_nosample", "DownBlockSpatioTemporal1D_nosample"),
                up_block_types=("UpBlockSpatioTemporal1D_nosample", "CrossAttnUpBlockSpatioTemporal1D_nosample", "CrossAttnUpBlockSpatioTemporal1D_nosample"),
                cross_attention_dim=1024,
                num_frames=num_frames,
                loss_type = loss_type,
                patch_size=2,
                need_attn=True,
            )
        ),
        # test用
        vae_weight_path = vae_weight_path,
        dit_weight_path = dit_weight_path
    ),
    optimizers=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=1e-2,
    ),
    schedulers=dict(
        name='constant',
        num_warmup_steps=0,
    ),
    train=dict(
        max_epochs=200,
        gradient_accumulation_steps=1,
        mixed_precision='no',
        checkpoint_interval=10,
        checkpoint_total_limit=10,
        log_with='tensorboard',
        log_interval=50,
        max_grad_norm=1.0,
        activation_checkpointing=True,
        save_ema=True,
        percentage=5/126,
    ),
    test=dict(
        mixed_precision='fp16',
        vae_save_dir=vae_save_dir,
        dit_save_dir=dit_save_dir,
        num_joints=num_joints,
    ),
)