import os
import random

import imageio
import numpy as np
from PIL import Image


def load_video(video_path, num_frames=0):
    reader = imageio.get_reader(video_path)
    if num_frames > 0:
        num_frames = min(reader.count_frames(), num_frames)
    else:
        num_frames = reader.count_frames()
    frames = []
    for i in range(num_frames):
        try:
            frame = Image.fromarray(reader.get_data(i))
            frames.append(frame)
        except Exception:
            continue
    return frames


def get_fps(video_path):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    return fps


def sample_frames(frames, sample_size=1, sample_stride=1, start_idx=None):
    frame_indexes = []
    for idx, frame in enumerate(frames):
        if frame is None or frame is False:
            continue
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
import torch
def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        frames = frames[indexes]
    else:
        assert False
    return frames

def add_music_to_video(src_video_path, dst_video_path):
    aud_path = src_video_path[:-4] + '.m4a'
    save_video_music_path = dst_video_path[:-4] + '_music' + dst_video_path[-4:]
    if os.path.exists(save_video_music_path):
        cmd = 'rm -f %s' % save_video_music_path
        os.system(cmd)
    cmd = 'ffmpeg -i %s -vn -y -acodec copy %s' % (src_video_path, aud_path)
    os.system(cmd)
    cmd = 'ffmpeg -i %s -i %s -vcodec copy -acodec copy %s' % (dst_video_path, aud_path, save_video_music_path)
    os.system(cmd)
    cmd = 'rm -f %s' % aud_path
    os.system(cmd)
