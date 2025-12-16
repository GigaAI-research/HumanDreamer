import numpy as np
from typing import Optional
import random
class DITTransform:
    def __init__(
        self,
        num_frames: int = 1,
        is_train: bool = False,
        num_joints: int = 18,  # body
        recon_part: str = 'whole',  # 'whole' or 'body' or 'face' or 'hand'
        train_mode: str = 'vae',
        mask_type: Optional[str] = None,
        stride: int = 1,  # 采样间隔
    ):
        self.is_train = is_train
        self.train_mode = train_mode
        self.mask_type = mask_type
        self.pose_transform = PoseTransform(
            num_frames=num_frames,
            is_train=is_train,
            num_joints=num_joints,
            recon_part=recon_part,
            train_mode=train_mode,
            mask_type=mask_type,
            stride=stride
        )

    def __call__(self, data_dict):
        data_dict = self.pose_transform(data_dict)
        if self.is_train:
            new_data_dict = {
                'pose_array': data_dict['pose_array'],
                'subset_array': data_dict['mask_array'],
            }
        else:
            new_data_dict = {
                'pose_array': data_dict['pose_array'],
                'subset_array': data_dict['mask_array'],
                'video_height': data_dict['video_height'],
                'video_width': data_dict['video_width'],
                # 'video': data_dict['video'],  # [f, h0, w0, 3]
            }
        if self.train_mode == 'dit':
            new_data_dict.update({'prompt': data_dict['prompt'], })
        if self.mask_type == 'score':
            new_data_dict.update({'score_array': data_dict['score_array']})
        # new_data_dict.update({'data_name': data_dict['data_name']})
        return new_data_dict

class PoseTransform:
    def __init__(
        self,
        num_frames: int = 1,
        is_train: bool = False,
        num_joints: int = 18,  # body
        recon_part: str = 'whole',  # 'whole' or 'body' or 'face' or 'hand'
        train_mode: str = 'vae',
        mask_type: Optional[str] = None,
        stride: int = 1,  # 采样间隔
    ):
        self.num_frames = num_frames
        self.is_train = is_train
        self.num_joints = num_joints
        self.recon_part = recon_part
        self.train_mode = train_mode
        self.mask_type = mask_type
        self.stride = stride
        
        
    def __call__(self, data_dict):
        new_data_dict = dict() 
        poses = data_dict['poses']
        poses_scores = data_dict['poses_scores']
        video_height = data_dict['video_height']
        video_width = data_dict['video_width']
        video_length = data_dict['video_length']
        # data_name = data_dict['tos_addr'] if 'tos_addr' in data_dict else data_dict['video_path']
        # new_data_dict.update({'data_name': data_name})
        if self.train_mode=='dit':
            prompt = data_dict['prompt']
            # prompt = 'A man is swinging the ax to chop wood in a forest.'

            new_data_dict.update({
                                  'prompt': prompt,
                                  })

        if 'video_valid_range' not in data_dict:
            video_valid_range = [0, video_length - 1]
        else:
            if data_dict['video_valid_range'] is None:
                video_valid_range = [0, video_length - 1]
            else:
                video_valid_range = data_dict['video_valid_range']  # [t1,t2]
        try:
            poses = poses[video_valid_range[0]:video_valid_range[1]+1]  # 切割掉转场
        except:
            print('video_valid_range:',video_valid_range)
            print('poses:',poses)
            print('data_dict:',data_dict)
            assert False
        # 按照stride采样
        batch_idx = sample_frames(poses, sample_size=self.num_frames, sample_stride=self.stride, start_idx=0)  # FIXME start_idx should be random
        poses = [poses[i] for i in batch_idx]

        poses_scores = [poses_scores[i] for i in batch_idx]
        
        '''
        读入的数据:
        pose格式:（可用来画图）
        body坐标[0-1],不会有-1;subset[0-17],中间的0表示无效点;
        hand和face坐标[-1,1],-1表示无效点,正常点在[0,1]内
        score格式:[0-1],face小于0.8视为无效,其他小于0.3视为无效;0:18body,24:92face,92:134hand
        
        处理后数据：
        pose_array: [f,128,2] hand和face中无效点置0.其他不变
        mask_array: [f,128,1] 无效点在subset中为0,其他为1
        score_array: [f,128,1]不需要操作,全部是0-1之间;0:18body,24:60hand,60:128face
        '''
        whole_pose_list = []
        score_list = []
        mask_list = []
        for pose, score in zip(poses, poses_scores):
            hand_pose = pose['hands'][0:2,:,:]  # 单人 [2,21,2]
            hand_pose = hand_pose.reshape(-1,2)  # [42,2]
            face_pose = pose['faces'][0]  # 单人 [68,2]
            body_pose = pose['bodies']['candidate'][0:self.num_joints]
            body_score = score[0, 0:18].reshape(-1,1)  # [18,1]
            hand_score = score[0, 92:134].reshape(-1,1)  # [42,1]
            face_score = score[0, 24:92].reshape(-1,1)  # [68,1]
            
            body_subset = pose['bodies']['subset'][0:1]  # [1,18] 单人,必须取0:18。后续如果多人,需要修改。而且需要注意不同帧的人数可能不同。
            body_subset = body_subset.transpose(1,0)  # [1,18] --> [18,1]
            body_mask = np.where(body_subset == -1, 0, 1)  # 将subset中-1的值替换为0,其他值替换为1
            face_mask = np.where(face_score<0.8, 0, 1)  # [68,1]
            hand_mask = np.where(hand_score<0.3, 0, 1)
            whole_mask = np.concatenate([body_mask,hand_mask,face_mask],axis=0)
            
            face_pose[face_score[:,0] < 0.8] = 0
            hand_pose[hand_score[:,0] < 0.3] = 0
            
            if self.recon_part == 'whole':
                whole_pose = np.concatenate([body_pose,hand_pose,face_pose],axis=0)
                whole_score = np.concatenate([body_score,hand_score,face_score],axis=0)
            else:
                assert False, 'recon_part must be whole'
            whole_pose_list.append(whole_pose)
            score_list.append(whole_score)
            mask_list.append(whole_mask)
        
        score_array = np.array(score_list)  # [f,128,1]
        mask_array = np.array(mask_list)  # [f,128,1]
        pose_array = np.array(whole_pose_list) # [f,128,2]
        
        if self.mask_type == 'score':
            new_data_dict.update({'score_array': score_array})
 
        rela_pose_array = abso_to_rela_new(pose_array) # [f,128,2]
        new_data_dict.update({'pose_array': rela_pose_array})

        if self.is_train:
            new_data_dict.update(
                {
                    'mask_array': mask_array,
                }
            )
        else:  # test
            # video = data_dict['video'] # <class 'decord.video_reader.VideoReader'>
            # video_data = np.array([video[i].asnumpy() for i in batch_idx])  # [f, h0, w0, 3]
            new_data_dict.update(
                {   
                    'video_height': video_height,
                    'video_width': video_width,
                    'mask_array': mask_array,
                    # 'video': video_data, # [f, h0, w0, 3]
                }
            )
        return new_data_dict


def abso_to_rela_new(abso_pos_array):
    # abso_pos_array: [f, num_points, 2]
    # rela_pos_array: [f, num_points, 2]
    num_points = abso_pos_array.shape[1]
    num_frames = abso_pos_array.shape[0]
    root_index = 1
    rela_pos_array = np.zeros_like(abso_pos_array)  # [f, num_points, 2]
    for t in range(num_frames):
        # Get absolute positions for this timestamp
        abso_pos = abso_pos_array[t]  # [num_points, 2]
        rela_pos = np.zeros((num_points, 2))  # [num_points, 2]
        rela_pos[root_index] = abso_pos[root_index]
        for i in range(num_points):
            if i == root_index:
                continue
            rela_pos[i] = abso_pos[i] - abso_pos[root_index]  # (xi,yi)
        rela_pos_array[t] = rela_pos
    assert rela_pos_array.shape == (num_frames, num_points, 2)
    return rela_pos_array  # [f, num_points, 2]


def get_prompt(data_dict):
    if 'prompt_sharegpt4video_action' in data_dict:
        prompt = data_dict['prompt_sharegpt4video_action']
    elif 'prompt_sharegpt4video' in data_dict:
        prompt = data_dict['prompt_sharegpt4video']
    elif 'prompt_panda' in data_dict:
        prompt = data_dict['prompt_panda']
    elif 'action' in data_dict:
        prompt = data_dict['action']
    elif 'prompt' in data_dict:
        prompt = data_dict['prompt']
    else:
        raise ValueError('prompt not found in data_dict')
    return prompt


def sample_frames(frames, sample_size=1, sample_stride=1, start_idx=None):
    frame_indexes = []
    for idx, frame in enumerate(frames):
        if frame is None or frame is False:
            assert False, 'frame is None or False'
        frame_indexes.append(idx)
    assert len(frame_indexes) >= sample_size > 0
    sample_length = min(len(frame_indexes), (sample_size - 1) * sample_stride + 1)
    if start_idx is None:
        start_idx = random.randint(0, len(frame_indexes) - sample_length)
    else:
        assert 0 <= start_idx <= len(frame_indexes) - sample_length
    sample_indexes = np.linspace(start_idx, start_idx + sample_length - 1, sample_size, dtype=int)
    frame_indexes = np.array(frame_indexes)
    frame_indexes = frame_indexes[sample_indexes]
    assert len(frame_indexes) == sample_size
    return frame_indexes
