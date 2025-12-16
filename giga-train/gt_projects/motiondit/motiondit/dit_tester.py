import cv2
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from einops import rearrange
import imageio
import pickle
import copy
import os
from giga_datasets import DefaultCollator, DefaultSampler, load_dataset, SelectSampler,CLIPTextTransform, TrainTestSampler
from diffusers.schedulers import EulerDiscreteScheduler
from giga_models import (
                         StableVideoDiffusionPOSE_IT2VPipeline1D, 
                         AutoencoderKLTemporalDecoder1d,
                         MotionDiT,
                         )
from giga_models import utils as gm_utils
from giga_models.pipelines.vision.keypoints.pipeline_dwpose import draw_poses

from giga_train import Tester
from .dit_transforms import DITTransform

class DITTester(Tester):
    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        # batch_size = data_config.batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        batch_size = data_config.batch_size_per_gpu
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform_cfg = data_config.transform
        transform_type = transform_cfg.pop('type')
        if transform_type == 'DITTransform':
            transform = DITTransform(**transform_cfg)
        else:
            assert False
        self.frame_num = data_config.frame_num
        dataset.set_transform(transform)
        sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
        sampler_type = sampler_cfg.pop('type')
        if sampler_type == 'DefaultSampler':
            sampler = DefaultSampler(dataset, batch_size=batch_size, **sampler_cfg)
        elif sampler_type == 'SelectSampler':
            sampler = SelectSampler(dataset, batch_size=batch_size, **sampler_cfg)
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
        )
        return dataloader


    def get_models(self, model_config):
        self.train_mode = model_config.get('train_mode', None)
        self.mask_type = model_config.get('mask_type', None)
        clip_path = 'Path to stabilityai/stable-diffusion-2-1-base on huggingface'
        self.text_transform = CLIPTextTransform(model_path=clip_path, device=self.device)
        if self.train_mode == 'vae':
            vae_cfg = model_config.get('vae', {})
            vae_dtype = vae_cfg.get('dtype', 'float32')
            vae_dtype = getattr(torch, vae_dtype)
            vae_hp = vae_cfg.get('hyperparameters', {})
            vae = AutoencoderKLTemporalDecoder1d(**vae_hp)
            vae.requires_grad_(False)
            vae.to(self.device, dtype=vae_dtype)
            vae_weight_path = model_config.get('vae_weight_path', None)
            print('loading vae_weight_path from {}'.format(vae_weight_path))
            vae_ckpt = torch.load(vae_weight_path, map_location='cpu')
            vae.load_state_dict(vae_ckpt, strict=True)

            return vae
        elif self.train_mode == 'dit':
            # VAE
            vae_cfg = model_config.get('vae', {})
            vae_dtype = vae_cfg.get('dtype', 'float32')
            vae_dtype = getattr(torch, vae_dtype)
            vae_hp = vae_cfg.get('hyperparameters', {})
            self.vae = AutoencoderKLTemporalDecoder1d(**vae_hp)
            self.vae.requires_grad_(False)
            self.vae.to(self.device, dtype=vae_dtype)

            #     vae.enable_gradient_checkpointing()
            vae_weight_path = model_config.get('vae_weight_path', None)  # 这是测试VAE效果用的
            print('loading vae checkpoint from {}'.format(vae_weight_path))
            vae_ckpt = torch.load(vae_weight_path, map_location='cpu')
            self.vae.load_state_dict(vae_ckpt, strict=True)                
            model_name = 'stabilityai/stable-video-diffusion-img2vid-xt-1-1'
            model_name = gm_utils.get_model_path(model_name)
            scheduler=EulerDiscreteScheduler.from_pretrained(model_name, subfolder='scheduler', local_files_only=True)

            torch_dtype = torch.float32
            # dit
            dit_weight_path = model_config.get('dit_weight_path', None)
            dit = MotionDiT.from_pretrained(
                    dit_weight_path,
                    # subfolder='dit',
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    torch_dtype=torch_dtype,
            )
            dit.requires_grad_(False)
            model = StableVideoDiffusionPOSE_IT2VPipeline1D(
                    dit=dit,
                    vae=self.vae,
                    scheduler=scheduler,
                    image_encoder=None,
                    feature_extractor=None)
            # cal params
            # pytorch_total_params = sum(p.numel() for p in dit.parameters())
            # trainable_pytorch_total_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
            # print('Total - ', pytorch_total_params)
            # print('Trainable - ', trainable_pytorch_total_params)
            self.gen_type = model_config.get('gen_type', 'it2m')  # 使用首帧text控制
            model.to(self.device)
            return model



    def get_schedulers(self, scheduler):
        scheduler = get_scheduler(
            name=scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=scheduler.num_warmup_steps * self.num_processes,
            num_training_steps=self.max_steps * self.num_processes,
        )
        return scheduler
    

    def test(self):
        if self.is_main_process:
            num_joints = self.kwargs.get('num_joints', 18)
            # eval_type = ['visual','metrics','gen_video','mm_metrics'] # 保存哪些结果：计算指标，可视化结果，用于后续生成视频
            eval_type = ['visual']
            # eval_type = ['gen_video']
            # eval_type = ['metrics']
            # eval_type = ['mm_metrics']
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.seed)
            if self.train_mode=='vae':
                save_dir = self.kwargs.get('vae_save_dir', None)
            elif self.train_mode=='dit':
                save_dir = self.kwargs.get('dit_save_dir', None)
            if eval_type == ['mm_metrics']:
                save_dir = os.path.join(save_dir,'mm_metrics')
            if eval_type == ['visual']:
                save_dir = os.path.join(save_dir,'vis')
            if eval_type == ['gen_video']:
                save_dir = os.path.join(save_dir,'gen_video')
            if eval_type == ['metrics']:
                save_dir = os.path.join(save_dir,'metrics')
            
            os.makedirs(save_dir, exist_ok=True)
            for i, batch_dict in enumerate(self.dataloader):
                if i > 10:
                    break
                # if i < 66:
                #     continue
                # if i not in [0,1,2,10,28,34,40,54,58,65,72,82]:# For K400_test
                #     continue
                subset_tensor = batch_dict['subset_array']  # [b,f,num_points,1]
                rela_pose_tensor = batch_dict['pose_array'].to(torch.float32)  # [b,f,num_points,c]  #TODO hardcode
                video_height = batch_dict['video_height']  # [b,]
                video_width = batch_dict['video_width']  # [b,]
                # video = batch_dict['video']  # [b, f, h0, w0, 3]
                prompt = batch_dict['prompt']  # [b,]
                batch_size, num_frames, num_points = rela_pose_tensor.shape[:3]
                self.num_points = num_points
                if self.train_mode == 'vae':
                    self.set_num_frames(num_frames, self.model)
                # 当batchsize>1时不支持visual,gen_video,mm_metrics
                if batch_size > 1:
                    assert eval_type==['metrics']
                print(f'Generating {i}th batch, batch_size: {batch_size}')

                # text transform
                # prompt_embeds = batch_dict['prompt_embeds']  # [b,1,1024]
                prompt_embeds = self.text_transform(batch_dict['prompt'],mode='after_pool',to_numpy=False)  # [b,1024]
                prompt_embeds = rearrange(prompt_embeds,'b c -> b 1 c')  # [b,1,1024]

                if self.mask_type == 'score':
                    score_tensor = batch_dict['score_array']
                    input_pose = torch.cat([rela_pose_tensor, score_tensor], dim=-1).to(torch.float32)  # [b,f,num_points,3]
                else:
                    input_pose = rela_pose_tensor.to(torch.float32)  # [b,f,num_points,2]
                gt_rela_pose_seq = copy.deepcopy(rela_pose_tensor)  # [b,f,num_points,2]

                if eval_type.__contains__('mm_metrics'):
                    num_videos_per_prompt = 32  # 一个prompt生成多少个样本
                else:
                    num_videos_per_prompt = 1

                if self.train_mode=='vae':
                    
                    input_pose = rearrange(input_pose,'b f n c -> (b f) c n')  # [b*f,2,num_points]
                    output_pose = self.model(input_pose, num_frames=num_frames).sample  # [b*f,2,num_points]
                    input_pose = rearrange(input_pose,'(b f) c n -> b f n c', b=batch_size)  # [b,f,num_points,2]
                    output_pose = rearrange(output_pose,'(b f) c n -> b f n c', b=batch_size)  # [b,f,num_points,2]
                    
                elif self.train_mode=='dit':
                    # input_pose = rearrange(input_pose,'b f n c -> (b f) c n')  # [b*f,c,num_points]
                    ref_pose = input_pose[:,0]  # [b,num_points,c]
                    ref_pose = rearrange(ref_pose,'b n c -> b c n')  # [b,c,num_points]
                    input_pose = rearrange(input_pose,'b f n c -> (b f) c n')  # [b*f,c,num_points]
                    output_pose = self.model(
                        ref_pose = ref_pose,
                        prompt_embeds = prompt_embeds,
                        num_points = num_points,
                        num_frames=self.frame_num,
                        fps = 7,
                        noise_aug_strength = 0.02,
                        decode_chunk_size = self.frame_num, #self.frame_num,#应当和vae的帧数相符合
                        video_chunk_size = self.frame_num,
                        generator=generator,
                        gen_type=self.gen_type,
                        num_videos_per_prompt=num_videos_per_prompt,
                    )  # [b,c,f,num_points]
                    
                    output_pose = rearrange(output_pose,'b c f n -> b f n c')  # [b,f,num_points,c]
                
                    
                output_pose = output_pose.detach().cpu().numpy()
                for b in range(batch_size):
                    output_pose_b = output_pose[b:num_videos_per_prompt+b]  # [num_videos_per_prompt,f,num_points,c]
                    self.eval_dict = {}
                    self.eval_dict['gen_poses_list'] = []
                    for n_v in range(num_videos_per_prompt):
                        if self.mask_type == 'score':
                            pred_motion = output_pose_b[n_v,:,:,:2]  # [f,num_points,2]  # TODO only support bsz=1
                            pred_subset = output_pose_b[n_v,:,:,2:3]  # [f,num_points,1]
                        else:
                            pred_motion = output_pose_b[n_v]  # [f,num_points,2]
                        # post processing
                        # rela to abso
                        abso_poses_seq = rela_to_abso_new(pred_motion)  # [f, num_points, 2]
                        gt_abso_poses_seq = rela_to_abso_new(gt_rela_pose_seq[b].cpu().numpy())  # [f, num_points, 2]

                        self.visualize_pose(
                                num_frames=num_frames,
                                pred_subset=pred_subset,
                                # video=video[b],
                                video = None,
                                prompt=prompt[b],
                                prompt_embeds=prompt_embeds[b:b+1],
                                abso_poses_seq=abso_poses_seq,
                                gt_abso_poses_seq=gt_abso_poses_seq,
                                num_joints=num_joints,
                                video_height=video_height[b],
                                video_width=video_width[b],
                                subset_tensor=subset_tensor[b:b+1],
                                save_dir=save_dir,
                                i=i,
                                eval_type=eval_type,
                                n_v=n_v  # 同一个prompt生成多个视频
                            )
                    if eval_type.__contains__('metrics') or eval_type.__contains__('mm_metrics'):
                        # pickle_path = os.path.join(save_dir, f'{i}_eval_dict.pkl')
                        pickle_path = os.path.join(save_dir, f'{i*batch_size+b}_eval_dict.pkl')
                        with open(pickle_path, 'wb') as f:
                            pickle.dump(self.eval_dict, f)
            
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

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
    
    def visualize_pose(self, num_frames, pred_subset, video, prompt, prompt_embeds, abso_poses_seq, gt_abso_poses_seq, num_joints, video_height, video_width, subset_tensor, save_dir, i, eval_type=['visual'], n_v=None):
        # draw a seq
        pose = {'bodies':{'candidate':[],'subset':[]},'hands':[],'faces':[]}
        gt_pose = {'bodies':{'candidate':[],'subset':[]},'hands':[],'faces':[]}
        for j in range(num_frames):
            ori_subset = np.linspace(0,num_joints-1,num_joints).astype(np.int32).reshape(-1,1) # [18,1]
            if self.mask_type == 'score':
                pred_mask = np.zeros((self.num_points,1))
                pred_mask[0:18] = np.where(pred_subset[j][0:18]<0.3,0,1)
                pred_mask[18:60] = np.where(pred_subset[j][18:60]<0.3,0,1)
                pred_mask[60:128] = np.where(pred_subset[j][60:128]<0.8,0,1)
            else: 
                pred_mask = np.ones((self.num_points,1))
            pred_body_mask = pred_mask[0:18]  # [18,1]
            pred_hand_mask = pred_mask[18:60]
            pred_face_mask = pred_mask[60:128]
            
            pred_body_pose = abso_poses_seq[j][0:18]  # [18, 2]
            pred_hand_pose = abso_poses_seq[j][18:60]  # [42,2]
            pred_hand_pose = np.where(pred_hand_mask==0,-1,pred_hand_pose)  # 将hands和face部分mask为0的点，坐标设为-1
            pred_face_pose = abso_poses_seq[j][60:128]  # [68,2]
            pred_face_pose = np.where(pred_face_mask==0,-1,pred_face_pose)  # 将hands和face部分mask为0的点，坐标设为-1
            
            pose['bodies']['candidate'] = pred_body_pose  # [18, 2]
            pose['hands'] = pred_hand_pose.reshape(2,21,2)  # [2,21,2]
            pose['faces'] = pred_face_pose.reshape(1,68,2)  # [1,68,2]
            gt_pose['bodies']['candidate'] = gt_abso_poses_seq[j][0:18]
            gt_pose['hands'] = gt_abso_poses_seq[j][18:60].reshape(2,21,2)
            gt_pose['faces'] = gt_abso_poses_seq[j][60:].reshape(1,68,2)
            gt_body_mask = subset_tensor[0][j][0:18].cpu().numpy().astype(np.int32) # [18,1]
            
            pose['bodies']['subset'] = np.where(pred_body_mask==0,-1,ori_subset).transpose(1,0) # [1,18]mask中值为0的，对应的subset中的值为-1；mask中值为1的，对应的subset中的值不变
            gt_pose['bodies']['subset'] = np.where(gt_body_mask==0,-1,ori_subset).transpose(1,0) # [1,18]mask中值为0的，对应的subset中的值为-1；mask中值为1的，对应的subset中的值不变
            assert pose['bodies']['subset'].shape[0]==1
            assert gt_pose['bodies']['subset'].shape[0]==1
            pose_image = draw_poses(pose, height=video_height, width=video_width, draw_body=True, draw_hand=True, draw_face=True)
            gt_pose_image = draw_poses(gt_pose, height=video_height, width=video_width, draw_body=True, draw_hand=True, draw_face=True)
            if j == 0:
                images = [pose_image]
                gt_images = [gt_pose_image]
                abso_poses_list = [copy.deepcopy(pose)]
                gt_poses_list = [copy.deepcopy(gt_pose)]
            else:
                images.append(pose_image)
                gt_images.append(gt_pose_image)
                abso_poses_list.append(copy.deepcopy(pose))
                gt_poses_list.append(copy.deepcopy(gt_pose))


        # video_array = video.cpu().numpy()  # [f,h,w,3]

        if eval_type.__contains__('gen_video'):
            # For Pose-to-Video
            first_frame = video_array[0]
            first_frame_path = os.path.join(save_dir, f'{i}_first_frame.jpg')# 保存首帧图像
            imageio.imwrite(first_frame_path, first_frame)
            pickle_path = os.path.join(save_dir, f'{i}_abso_poses_list.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(abso_poses_list, f)
            pickle_path = os.path.join(save_dir, f'{i}_gt_poses_list.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(gt_poses_list, f)
            with open(os.path.join(save_dir, f'{i}_prompt.txt'), 'w') as f:
                f.write(prompt)
        if eval_type.__contains__('metrics') or eval_type.__contains__('mm_metrics'):
            # For Eval
            # 保存abso_poses_list为pkl
            # eval_dict = {}
            self.eval_dict['gen_poses_list'].append(abso_poses_list)
            self.eval_dict['gt_poses_list'] = gt_poses_list
            self.eval_dict['prompt'] = prompt
            self.eval_dict['prompt_embeds'] = prompt_embeds[0].detach().cpu().numpy() # [1,1024]
        if eval_type.__contains__('visual'):
            images_array = np.array([np.array(img) for img in images])  # [f,h,w,3]
            gt_images_array = np.array([np.array(img) for img in gt_images])  # [f,h,w,3]
            output_images_array = np.concatenate([gt_images_array,images_array],axis=2)  # [f,h,w*2,3]
            # output_images_array = images_array
            # 在视频里写上prompt
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = prompt
            org = (20, 20)
            fontScale = 0.5
            color = (255, 255, 255)
            thickness = 1
            for k in range(output_images_array.shape[0]):
                image = output_images_array[k]
                image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                output_images_array[k] = image
            if len(prompt)>50:
                prompt = prompt[:50]
            print('prompt:',prompt)
            output_video_path = os.path.join(save_dir, f'{i}_{n_v}_{prompt}.mp4')
            imageio.mimwrite(output_video_path, output_images_array, 'MP4', fps=15)


def rela_to_abso_new(rela_pos):
    # rela_pos: [f, num_points, 2]
    # abso_pos: [f, num_points, 2]
    num_frames, num_points = rela_pos.shape[:2]
    assert rela_pos.shape[2] == 2
    root_index = 1
    abso_pos_array = np.zeros_like(rela_pos)
    for t in range(num_frames):
        # Get relative positions for this timestamp
        positions = rela_pos[t]#.reshape(num_points, 2)  # (num_points, 2)
        abso_pos = np.zeros((num_points, 2))
        abso_pos[root_index] = positions[root_index]  # root节点坐标
        for i in range(num_points):
            if i == root_index:
                continue
            abso_pos[i] = abso_pos[root_index] + positions[i]
        abso_pos_array[t] = abso_pos
    assert abso_pos_array.shape == (num_frames, num_points, 2)
    return abso_pos_array  # (f, num_points, 2)

