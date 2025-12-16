import copy

import cv2
import numpy as np
import torch
from controlnet_aux import DWposeDetector as _DWposeDetector
from controlnet_aux.dwpose import HWC3, resize_image, util
from PIL import Image

from ...pipeline import BasePipeline

KEYPOINT_INDEXES = {
    'nose': 0,
    'neck': 1,
    'shoulder_left': 2,
    'wrist_left': 3,
    'hand_left': 4,
    'shoulder_right': 5,
    'wrist_right': 6,
    'hand_right': 7,
    'waist_left': 8,
    'knee_left': 9,
    'foot_left': 10,
    'waist_right': 11,
    'knee_right': 12,
    'foot_right': 13,
    'eye_left': 14,
    'eye_right': 15,
    'ear_left': 16,
    'ear_right': 17,
}


class DWposePipeline(BasePipeline):
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None):
        self.model = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image, include_body=True, include_hand=False, include_face=False, return_image=True):
        output = self.model(
            image,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            return_image=return_image,
        )
        return output


class DWposeDetector(_DWposeDetector):
    def __call__(
        self,
        input_image,
        detect_resolution=512,
        include_body=True,
        include_hand=False,
        include_face=False,
        return_image=True,
        *kwargs,
    ):
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        input_image = resize_image(input_image, detect_resolution)
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(input_image.shape[1])
        candidate[..., 1] /= float(input_image.shape[0])
        body = candidate[:, :18].copy()
        body = body.reshape(nums * 18, locs)
        score = subset[:, :18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1
        un_visible = subset < 0.3
        candidate[un_visible] = -1
        # foot = candidate[:,18:24]
        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])
        bodies = dict(candidate=body, subset=score)
        poses = dict()
        if include_body:
            poses.update(bodies=bodies)
        if include_hand:
            poses.update(hands=hands)
        if include_face:
            poses.update(faces=faces)
        if return_image:
            detected_map = draw_poses(
                poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face
            )
            return detected_map
        else:
            return poses


def draw_poses(poses, height, width, draw_body=True, draw_hand=False, draw_face=False):
    canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    if draw_body:
        bodies = poses['bodies']
        candidate = bodies['candidate']
        subset = bodies['subset']
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if draw_hand:
        hands = poses['hands']
        canvas = util.draw_handpose(canvas, hands)
    if draw_face:
        faces = poses['faces']
        canvas = util.draw_facepose(canvas, faces)
    canvas = Image.fromarray(canvas)
    return canvas


def preprocess_ref_poses(poses, height, width):
    radio = float(width) / height
    if height < 256 or width < 256:
        return '当前图像分辨率过低，请重新上传'
    if radio < 0.4 or radio > 2.5:
        return '当前图像尺寸不符合要求，请重新上传'
    boxes = get_boxes(poses, height, width, include_body=True, include_hand=True, include_face=True)
    valid_indexes = []
    with_big_pose = False
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx]
        box_h = y2 - y1
        if box_h / height > 0.5:
            if not _check_face(poses, idx):
                return '存在严重侧脸或人脸不完整，请重新上传'
            if not _check_body(poses, idx):
                return '存在严重人体侧身或人体不完整，请重新上传'
            valid_indexes.append(idx)
            with_big_pose = True
        elif box_h / height > 0.25:
            if not _check_face(poses, idx):
                continue
            if not _check_body(poses, idx):
                continue
            valid_indexes.append(idx)
    if len(valid_indexes) == 0 or not with_big_pose:
        return '未检测到人体或人体过小，请重新上传'
    return get_poses(poses, valid_indexes)


def preprocess_poses(poses_list, num_poses=1, align_idx=0):
    new_poses_list = []
    for i, poses in enumerate(poses_list):
        new_poses = filter_poses(poses, area_mode='area', max_poses=num_poses)
        if get_num_poses(new_poses) != num_poses:
            return '每帧的人体数量不相等'
        new_poses_list.append(new_poses)
    for idx in range(num_poses):
        if not _check_face(new_poses_list[align_idx], idx):
            return '存在严重侧脸或人脸不完整'
        if not _check_body(new_poses_list[align_idx], idx):
            return '存在严重人体侧身或人体不完整'
    return new_poses_list


def get_num_poses(poses):
    num_poses = None
    if 'bodies' in poses:
        num_poses = len(poses['bodies']['subset'])
    if 'hands' in poses:
        if num_poses is None:
            num_poses = len(poses['hands']) // 2
        else:
            assert num_poses == len(poses['hands']) // 2
    if 'faces' in poses:
        if num_poses is None:
            num_poses = len(poses['faces'])
        else:
            assert num_poses == len(poses['faces'])
    assert num_poses is not None
    return num_poses


def get_boxes(poses, height, width, include_body=True, include_hand=False, include_face=False):
    num_poses = get_num_poses(poses)
    boxes = np.zeros((num_poses, 4), dtype=np.float32)
    for i in range(num_poses):
        points = []
        if include_body:
            bodies = poses['bodies']
            candidate = bodies['candidate']
            subset = bodies['subset'][i]
            subset = subset[subset >= 0].astype(np.int32)
            candidate = candidate[subset]
            points.append(candidate)
        if include_hand:
            hands = poses['hands'].reshape((2, num_poses, -1, 2))
            hands = hands[:, i].reshape((-1, 2))
            hands = hands[hands.sum(-1) >= 0]
            points.append(hands)
        if include_face:
            faces = poses['faces'][i]
            faces = faces[faces.sum(-1) >= 0]
            points.append(faces)
        points = np.concatenate(points, axis=0)
        if len(points) > 0:
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1 = max(min(x1 * width, width - 1), 0)
            y1 = max(min(y1 * height, height - 1), 0)
            x2 = max(min(x2 * width, width - 1), 0)
            y2 = max(min(y2 * height, height - 1), 0)
            boxes[i] = [x1, y1, x2, y2]
        else:
            boxes[i] = [0, 0, width - 1, height - 1]
    return boxes


def get_face_centers(poses, height, width):
    num_poses = get_num_poses(poses)
    face_centers = np.zeros((num_poses, 2), dtype=np.float32)
    for i in range(num_poses):
        points = []
        if 'bodies' in poses:
            bodies = poses['bodies']
            candidate = bodies['candidate']
            subset = bodies['subset'][i, [0]]
            subset = subset[subset >= 0].astype(np.int32)
            candidate = candidate[subset]
            points.append(candidate)
        if 'faces' in poses:
            faces = poses['faces'][i, 27:]
            faces = faces[faces.sum(-1) >= 0]
            points.append(faces)
        points = np.concatenate(points, axis=0)
        if len(points) > 0:
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1 = max(min(x1 * width, width - 1), 0)
            y1 = max(min(y1 * height, height - 1), 0)
            x2 = max(min(x2 * width, width - 1), 0)
            y2 = max(min(y2 * height, height - 1), 0)
            ctr_x = (x1 + x2) / 2
            ctr_y = (y1 + y2) / 2
            face_centers[i] = [ctr_x, ctr_y]
        else:
            face_centers[i] = [-1, -1]
    return face_centers


def get_poses(poses, idx):
    idx_list = idx if isinstance(idx, list) else [idx]
    poses = copy.deepcopy(poses)
    if 'bodies' in poses:
        bodies = poses['bodies']
        candidate = bodies['candidate'].reshape((-1, 18, 2))
        subset = bodies['subset']
        candidate = candidate[idx_list]
        candidate = candidate.reshape((-1, 2))
        subset = subset[idx_list].reshape(-1)
        new_subset = np.arange(subset.shape[0], dtype=subset.dtype)
        new_subset[subset < 0] = subset[subset < 0]
        new_subset = new_subset.reshape((-1, 18))
        poses['bodies']['candidate'] = candidate
        poses['bodies']['subset'] = new_subset
    if 'hands' in poses:
        hands = poses['hands'].reshape((2, len(poses['hands']) // 2, -1, 2))
        hands = hands[:, idx_list]
        hands = hands.reshape((-1, poses['hands'].shape[1], 2))
        poses['hands'] = hands
    if 'faces' in poses:
        faces = poses['faces']
        poses['faces'] = faces[idx_list]
    return poses


def merge_poses(poses_list):
    if len(poses_list) == 1:
        return poses_list[0]
    new_poses = dict()
    if 'bodies' in poses_list[0]:
        candidate_list = [poses['bodies']['candidate'] for poses in poses_list]
        subset_list = [poses['bodies']['subset'] for poses in poses_list]
        candidate = np.concatenate(candidate_list, axis=0)
        subset = np.concatenate(subset_list, axis=0).reshape(-1)
        new_subset = np.arange(subset.shape[0], dtype=subset.dtype)
        new_subset[subset < 0] = subset[subset < 0]
        new_subset = new_subset.reshape((-1, 18))
        new_poses['bodies'] = {
            'candidate': candidate,
            'subset': new_subset,
        }
    if 'hands' in poses_list[0]:
        hands_list = [poses['hands'] for poses in poses_list]
        hands = np.concatenate(hands_list, axis=1)
        hands = hands.reshape((2 * len(hands_list), -1, 2))
        new_poses['hands'] = hands
    if 'faces' in poses_list[0]:
        faces_list = [poses['faces'] for poses in poses_list]
        faces = np.concatenate(faces_list, axis=0)
        new_poses['faces'] = faces
    return new_poses


def merge_pose_images(pose_images):
    if len(pose_images) == 1:
        return pose_images[0]
    new_pose_image = np.zeros((pose_images[0].height, pose_images[0].width, 3), dtype=np.uint8)
    for pose_image in pose_images:
        pose_image = np.array(pose_image)
        new_pose_image[pose_image > 0] = pose_image[pose_image > 0]
    new_pose_image = Image.fromarray(new_pose_image)
    return new_pose_image


def filter_poses(poses, area_mode='area', min_area=None, max_area=None, max_poses=None):
    num_poses = get_num_poses(poses)
    areas = []
    for i in range(num_poses):
        points = []
        if 'bodies' in poses:
            bodies = poses['bodies']
            candidate = bodies['candidate']
            subset = bodies['subset'][i]
            subset = subset[subset >= 0].astype(np.int32)
            candidate = candidate[subset]
            points.append(candidate)
        if 'hands' in poses:
            hands = poses['hands'].reshape((2, num_poses, -1, 2))
            hands = hands[:, i].reshape((-1, 2))
            hands = hands[hands.sum(-1) >= 0]
            points.append(hands)
        if 'faces' in poses:
            faces = poses['faces'][i]
            faces = faces[faces.sum(-1) >= 0]
            points.append(faces)
        points = np.concatenate(points, axis=0)
        if len(points) > 0:
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1 = max(min(x1, 1), 0)
            y1 = max(min(y1, 1), 0)
            x2 = max(min(x2, 1), 0)
            y2 = max(min(y2, 1), 0)
            if area_mode == 'area':
                areas.append((x2 - x1) * (y2 - y1))
            elif area_mode == 'height':
                areas.append(y2 - y1)
            elif area_mode == 'width':
                areas.append(x2 - x1)
            else:
                assert False
        else:
            areas.append(0)
    indexes = np.array(areas).argsort()[::-1]
    idx_list = []
    for idx in indexes:
        if min_area is not None and areas[idx] < min_area:
            continue
        if max_area is not None and areas[idx] > max_area:
            continue
        idx_list.append(idx)
    if max_poses is not None:
        idx_list = idx_list[:max_poses]
    return get_poses(poses, idx_list)


def align_poses(
    ref_image,
    ref_poses,
    poses_list,
    pose_image_height,
    pose_image_width,
    include_body=True,
    include_hand=False,
    include_face=False,
    align_idx=0,
    method=1,
    crop_ref_image=True,
):
    ref_poses, poses_list = _process_poses(
        ref_poses=ref_poses,
        poses_list=poses_list,
        ref_image_height=ref_image.height,
        ref_image_width=ref_image.width,
        pose_image_height=pose_image_height,
        pose_image_width=pose_image_width,
    )
    all_pose_images = []
    ref_height_list = []
    all_poses_list = []
    num_poses = get_num_poses(ref_poses)
    for i in range(num_poses):
        ref_poses_i = get_poses(ref_poses, i)
        poses_list_i = [get_poses(poses, i) for poses in poses_list]
        pose_images, ref_height, poses_list_i = _align_poses_single(
            ref_image=ref_image,
            ref_poses=ref_poses_i,
            poses_list=poses_list_i,
            pose_image_height=pose_image_height,
            pose_image_width=pose_image_width,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            align_idx=align_idx,
            method=method,
        )
        all_pose_images.append(pose_images)
        ref_height_list.append(ref_height)
        all_poses_list.append(poses_list_i)
    pose_images = [merge_pose_images(_) for _ in zip(*all_pose_images)]
    poses_list = [merge_poses(_) for _ in zip(*all_poses_list)]
    ref_height = max(ref_height_list)
    if crop_ref_image and ref_height > 0:
        crop_box = (0, 0, ref_image.width, int(ref_height))
        ref_image = ref_image.crop(crop_box)
        pose_images = [pose_image.crop(crop_box) for pose_image in pose_images]
    assert ref_image.height == pose_images[0].height and ref_image.width == pose_images[0].width
    return ref_image, pose_images, ref_poses, poses_list


def _process_poses(
    ref_poses,
    poses_list,
    ref_image_height,
    ref_image_width,
    pose_image_height,
    pose_image_width,
):
    ref_boxes = get_boxes(ref_poses, ref_image_height, ref_image_width)
    boxes_list = [get_boxes(poses, pose_image_height, pose_image_width) for poses in poses_list]
    ref_indexes = list(np.argsort(ref_boxes[:, 0]))
    indexes_list = [list(np.argsort(boxes[:, 0])) for boxes in boxes_list]
    for i, indexes in enumerate(indexes_list):
        if len(indexes) > len(ref_indexes):
            indexes = indexes[: len(ref_indexes)]
        elif len(indexes) < len(ref_indexes):
            extra = len(ref_indexes) - len(indexes)
            for j in range(extra):
                indexes.append(indexes[-1])
        indexes_list[i] = indexes
    ref_poses = get_poses(ref_poses, ref_indexes)
    poses_list = [get_poses(poses, indexes) for poses, indexes in zip(poses_list, indexes_list)]
    return ref_poses, poses_list


def _align_poses_single(
    ref_image,
    ref_poses,
    poses_list,
    pose_image_height,
    pose_image_width,
    include_body=True,
    include_hand=False,
    include_face=False,
    align_idx=0,
    method=1,
):
    ref_poses = copy.deepcopy(ref_poses)
    poses_list = copy.deepcopy(poses_list)
    _set_vis_indexes([ref_poses])
    _set_vis_indexes(poses_list)
    _scale_offset_poses([ref_poses], scale=(ref_image.width, ref_image.height))
    _scale_offset_poses(poses_list, scale=(pose_image_width, pose_image_height))
    scale = _get_body_scale(ref_poses, poses_list[align_idx])
    _scale_offset_poses(poses_list, scale=scale)
    offset = _get_body_offset(ref_poses, poses_list[align_idx])
    _scale_offset_poses(poses_list, offset=offset)
    if method == 1:
        face_scale = _get_face_scale(ref_poses, poses_list[align_idx])
        # hand_scale = _get_hand_scale(ref_poses, poses_list[align_idx])
        body_scales = _get_body_detail_scales(ref_poses, poses_list[align_idx], face_scale)
        for poses in poses_list:
            assert poses['bodies']['subset'][0][0] == 0 and poses['bodies']['subset'][0][1] == 1
            nose_coord = poses['bodies']['candidate'][0]
            neck_coord = _compute_coord(body_scales, poses, ['nose', 'neck'], nose_coord)
            shoulder_left_coord = _compute_coord(body_scales, poses, ['neck', 'shoulder_left'], neck_coord)
            wrist_left_coord = _compute_coord(body_scales, poses, ['shoulder_left', 'wrist_left'], shoulder_left_coord)
            hand_left_coord = _compute_coord(body_scales, poses, ['wrist_left', 'hand_left'], wrist_left_coord)
            shoulder_right_coord = _compute_coord(body_scales, poses, ['neck', 'shoulder_right'], neck_coord)
            wrist_right_coord = _compute_coord(
                body_scales, poses, ['shoulder_right', 'wrist_right'], shoulder_right_coord
            )
            hand_right_coord = _compute_coord(body_scales, poses, ['wrist_right', 'hand_right'], wrist_right_coord)
            waist_left_coord = _compute_coord(body_scales, poses, ['neck', 'waist_left'], neck_coord)
            knee_left_coord = _compute_coord(body_scales, poses, ['waist_left', 'knee_left'], waist_left_coord)
            foot_left_coord = _compute_coord(body_scales, poses, ['knee_left', 'foot_left'], knee_left_coord)
            waist_right_coord = _compute_coord(body_scales, poses, ['neck', 'waist_right'], neck_coord)
            knee_right_coord = _compute_coord(body_scales, poses, ['waist_right', 'knee_right'], waist_right_coord)
            foot_right_coord = _compute_coord(body_scales, poses, ['knee_right', 'foot_right'], knee_right_coord)
            # process faces
            faces_in_body = poses['bodies']['candidate'][[-4, -3, -2, -1, 0]]
            vis_indexes_in_body = [poses['vis_indexes']['bodies'][i] for i in [-4, -3, -2, -1, 0]]
            faces = np.concatenate([poses['faces'][0], faces_in_body], axis=0)
            vis_indexes = poses['vis_indexes']['faces'] + vis_indexes_in_body
            faces[vis_indexes] *= face_scale
            faces[vis_indexes] += nose_coord - faces[-1]
            faces_in_body = faces[-5:-1, :]
            poses['faces'] = faces[:-5][None]
            # process hands
            hands_list = []
            for i in range(2):
                if i == 0:
                    if poses['vis_indexes']['bodies'][7]:
                        hand_idx = 7
                        hand_coord = hand_right_coord
                    elif poses['vis_indexes']['bodies'][6]:
                        hand_idx = 6
                        hand_coord = wrist_right_coord
                    else:
                        hand_idx = -1
                else:
                    if poses['vis_indexes']['bodies'][4]:
                        hand_idx = 4
                        hand_coord = hand_left_coord
                    elif poses['vis_indexes']['bodies'][3]:
                        hand_idx = 3
                        hand_coord = wrist_left_coord
                    else:
                        hand_idx = -1
                hands = poses['hands'][i]
                vis_indexes = poses['vis_indexes']['hands'][i]
                if hand_idx > 0:
                    hands[vis_indexes] += hand_coord - poses['bodies']['candidate'][hand_idx : hand_idx + 1]
                else:
                    hands[:] = -1
                    for k in range(len(vis_indexes)):
                        vis_indexes[k] = False
                hands_list.append(hands)
            poses['hands'] = np.stack(hands_list, axis=0)
            # process bodies
            bodies = [
                nose_coord,
                neck_coord,
                shoulder_left_coord,
                wrist_left_coord,
                hand_left_coord,
                shoulder_right_coord,
                wrist_right_coord,
                hand_right_coord,
                waist_left_coord,
                knee_left_coord,
                foot_left_coord,
                waist_right_coord,
                knee_right_coord,
                foot_right_coord,
            ] + faces_in_body.tolist()
            bodies = np.array(bodies)
            poses['bodies']['candidate'] = bodies
    else:
        assert method == 0
    offset = _get_ground_offset(ref_poses, poses_list[align_idx])
    _scale_offset_poses(poses_list, offset=offset)
    # pose_images
    pose_images = _draw_pose_images(
        poses_list,
        height=ref_image.height,
        width=ref_image.width,
        draw_body=include_body,
        draw_hand=include_hand,
        draw_face=include_face,
    )
    # crop ref_image
    ref_vis_indexes = ref_poses['vis_indexes']['bodies']
    tem_vis_indexes = _union_vis_indexes([poses['vis_indexes']['bodies'] for poses in poses_list])
    if sum(ref_vis_indexes) > sum(tem_vis_indexes):
        cross_indexes = _intersect_vis_indexes([ref_vis_indexes, tem_vis_indexes])
        vis_bodies = ref_poses['bodies']['candidate'][cross_indexes]
        ref_height = np.max(vis_bodies, axis=0)[1]
    else:
        ref_height = -1
    _scale_offset_poses(poses_list, scale=(1.0 / ref_image.width, 1.0 / ref_image.height))
    _set_vis_indexes(poses_list)
    _clip_poses(poses_list)
    for poses in poses_list:
        poses.pop('vis_indexes')
    return pose_images, ref_height, poses_list


def _point_in_image(point):
    if point[0] < 0 or point[1] < 0 or point[0] >= 1 or point[1] >= 1:
        return False
    return True


def _set_vis_indexes(poses_list):
    for poses in poses_list:
        poses['vis_indexes'] = dict()
        if 'bodies' in poses:
            candidate = poses['bodies']['candidate']
            subset = poses['bodies']['subset'][0]
            vis_indexes = [True] * subset.shape[0]
            for idx, state in enumerate(subset):
                if state < 0 or not _point_in_image(candidate[idx]):
                    vis_indexes[idx] = False
            poses['vis_indexes']['bodies'] = vis_indexes
        if 'hands' in poses:
            hands = poses['hands']
            vis_indexes = [[True] * hands.shape[1]] * hands.shape[0]
            for i in range(hands.shape[0]):
                for idx, point in enumerate(hands[i]):
                    if not _point_in_image(point):
                        vis_indexes[i][idx] = False
            poses['vis_indexes']['hands'] = vis_indexes
        if 'faces' in poses:
            faces = poses['faces'][0]
            vis_indexes = [True] * faces.shape[0]
            for idx, point in enumerate(faces):
                if not _point_in_image(point):
                    vis_indexes[idx] = False
            poses['vis_indexes']['faces'] = vis_indexes


def _intersect_vis_indexes(indexes_list):
    vis_indexes = [True] * len(indexes_list[0])
    for idx in range(len(vis_indexes)):
        for i in range(len(indexes_list)):
            if not indexes_list[i][idx]:
                vis_indexes[idx] = False
                break
    return vis_indexes


def _union_vis_indexes(indexes_list):
    vis_indexes = [False] * len(indexes_list[0])
    for idx in range(len(vis_indexes)):
        for i in range(len(indexes_list)):
            if indexes_list[i][idx]:
                vis_indexes[idx] = True
                break
    return vis_indexes


def _scale_offset_poses(poses_list, scale=None, offset=None):
    for poses in poses_list:
        vis_indexes = poses['vis_indexes']
        if 'bodies' in poses:
            candidate = poses['bodies']['candidate'][vis_indexes['bodies']].copy()
            if scale is not None:
                candidate *= scale
            if offset is not None:
                candidate += offset
            poses['bodies']['candidate'][vis_indexes['bodies']] = candidate
        if 'hands' in poses:
            hands = poses['hands']
            for i in range(hands.shape[0]):
                hands_i = hands[i][vis_indexes['hands'][i]].copy()
                if scale is not None:
                    hands_i *= scale
                if offset is not None:
                    hands_i += offset
                poses['hands'][i][vis_indexes['hands'][i]] = hands_i
        if 'faces' in poses:
            faces = poses['faces'][0][vis_indexes['faces']].copy()
            if scale is not None:
                faces *= scale
            if offset is not None:
                faces += offset
            poses['faces'][0][vis_indexes['faces']] = faces


def _clip_poses(poses_list):
    for poses in poses_list:
        vis_indexes = poses['vis_indexes']
        if 'bodies' in poses:
            candidate = poses['bodies']['candidate'][vis_indexes['bodies']].copy()
            subset = poses['bodies']['subset'][0][vis_indexes['bodies']].copy()
            poses['bodies']['candidate'][:] = -1
            poses['bodies']['subset'][0][:] = -1
            poses['bodies']['candidate'][vis_indexes['bodies']] = candidate
            poses['bodies']['subset'][0][vis_indexes['bodies']] = subset
        if 'hands' in poses:
            hands = poses['hands']
            for i in range(hands.shape[0]):
                hands_i = hands[i][vis_indexes['hands'][i]].copy()
                poses['hands'][i][:] = -1
                poses['hands'][i][vis_indexes['hands'][i]] = hands_i
        if 'faces' in poses:
            faces = poses['faces'][0][vis_indexes['faces']].copy()
            poses['faces'][0][:] = -1
            poses['faces'][0][vis_indexes['faces']] = faces


def _get_body_scale(ref_poses, poses):
    assert ref_poses['bodies']['subset'][0][2] == 2 and ref_poses['bodies']['subset'][0][5] == 5
    assert poses['bodies']['subset'][0][2] == 2 and poses['bodies']['subset'][0][5] == 5
    # scale_w
    left_shoulder, right_shoulder = ref_poses['bodies']['candidate'][2], ref_poses['bodies']['candidate'][5]
    ref_w = np.linalg.norm(left_shoulder - right_shoulder)
    left_shoulder, right_shoulder = poses['bodies']['candidate'][2], poses['bodies']['candidate'][5]
    aln_w = np.linalg.norm(left_shoulder - right_shoulder)
    scale_w = ref_w / aln_w
    # scale_h
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['bodies'], poses['vis_indexes']['bodies']])
    ignore_indexes = [3, 4, 6, 7]  # ignore hands
    for ignore_index in ignore_indexes:
        cross_indexes[ignore_index] = False
    assert sum(cross_indexes) > 0
    vis_bodies = ref_poses['bodies']['candidate'][cross_indexes]
    h_min = np.min(vis_bodies, axis=0)[1]
    h_max = np.max(vis_bodies, axis=0)[1]
    ref_h = h_max - h_min
    vis_bodies = poses['bodies']['candidate'][cross_indexes]
    h_min = np.min(vis_bodies, axis=0)[1]
    h_max = np.max(vis_bodies, axis=0)[1]
    aln_h = h_max - h_min
    scale_h = ref_h / aln_h
    return scale_w, scale_h


def _get_body_offset(ref_poses, poses):
    assert ref_poses['bodies']['subset'][0][0] == 0
    assert poses['bodies']['subset'][0][0] == 0
    ref_coord = ref_poses['bodies']['candidate'][0]
    aln_coord = poses['bodies']['candidate'][0]
    offset_w = int(ref_coord[0] - aln_coord[0])
    offset_h = int(ref_coord[1] - aln_coord[1])
    return offset_w, offset_h


def _get_face_scale(ref_poses, poses):
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['faces'], poses['vis_indexes']['faces']])
    assert sum(cross_indexes) > 0
    ref_w, ref_h = _get_wh(ref_poses['faces'][0][cross_indexes])
    aln_w, aln_h = _get_wh(poses['faces'][0][cross_indexes])
    assert ref_w > 0 and aln_w > 0 and ref_h > 0 and aln_h > 0
    scale_w = ref_w / aln_w
    scale_h = ref_h / aln_h
    return scale_w, scale_h


def _get_hand_scale(ref_poses, poses):
    idx = 0
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['hands'][0], poses['vis_indexes']['hands'][0]])
    cross_indexes_2 = _intersect_vis_indexes([ref_poses['vis_indexes']['hands'][1], poses['vis_indexes']['hands'][1]])
    if sum(cross_indexes_2) > sum(cross_indexes):
        idx = 1
        cross_indexes = cross_indexes_2
    assert sum(cross_indexes) > 0
    ref_w, ref_h = _get_wh(ref_poses['hands'][idx][cross_indexes])
    aln_w, aln_h = _get_wh(poses['hands'][idx][cross_indexes])
    assert ref_w > 0 and aln_w > 0 and ref_h > 0 and aln_h > 0
    scale_w = ref_w / aln_w
    scale_h = ref_h / aln_h
    return scale_w, scale_h


def _get_body_detail_scales(ref_poses, poses, face_scale):
    all_indexes = [
        [['nose', 'neck']],
        [['neck', 'shoulder_left'], ['neck', 'shoulder_right']],
        [['neck', 'waist_left'], ['neck', 'waist_right']],
        [['waist_left', 'knee_left'], ['waist_right', 'knee_right']],
        [['knee_left', 'foot_left'], ['knee_right', 'foot_right']],
    ]
    all_indexes_2 = [
        [['shoulder_left', 'wrist_left'], ['shoulder_right', 'wrist_right']],
        [['wrist_left', 'hand_left'], ['wrist_right', 'hand_right']],
    ]
    all_scales = dict()
    ref_candidate = ref_poses['bodies']['candidate']
    ref_vis_indexes = ref_poses['vis_indexes']['bodies']
    candidate = poses['bodies']['candidate']
    vis_indexes = poses['vis_indexes']['bodies']
    for indexes in all_indexes:
        scales = []
        ref_dist_max = -10000
        aln_dist_max = -10000
        for indexes_i in indexes:
            src_idx = KEYPOINT_INDEXES[indexes_i[0]]
            dst_idx = KEYPOINT_INDEXES[indexes_i[1]]
            if ref_vis_indexes[src_idx] and ref_vis_indexes[dst_idx] and vis_indexes[src_idx] and vis_indexes[dst_idx]:
                ref_dist = np.linalg.norm(ref_candidate[src_idx] - ref_candidate[dst_idx])
                aln_dist = np.linalg.norm(candidate[src_idx] - candidate[dst_idx])
                ref_dist_max = max(ref_dist, ref_dist_max)
                aln_dist_max = max(aln_dist, aln_dist_max)
                scales.append(ref_dist / aln_dist)
            else:
                scales.append(-1)
        for i in range(len(scales)):
            if scales[i] != -1:
                scales[i] = ref_dist_max / aln_dist_max
        for i, indexes_i in enumerate(indexes):
            name = '_'.join(indexes_i)
            assert name not in all_scales
            all_scales[name] = scales[i]
    for indexes in all_indexes_2:
        for i, indexes_i in enumerate(indexes):
            name = '_'.join(indexes_i)
            assert name not in all_scales
            all_scales[name] = all_scales['neck_waist_left']
    nose_neck_scale = all_scales['nose_neck']
    if face_scale[1] < 0.9:
        nose_neck_scale = min(nose_neck_scale, 1.2)
    all_scales['nose_neck'] = nose_neck_scale
    return all_scales


def _compute_coord(body_scales, poses, indexes, base_coord):
    src_idx = KEYPOINT_INDEXES[indexes[0]]
    dst_idx = KEYPOINT_INDEXES[indexes[1]]
    candidate = poses['bodies']['candidate']
    scale = body_scales['_'.join(indexes)]
    coord = candidate[dst_idx] - candidate[src_idx]
    if scale > 0:
        coord *= scale
    coord += base_coord
    return coord


def _get_ground_offset(ref_poses, poses):
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['bodies'], poses['vis_indexes']['bodies']])
    ignore_indexes = [3, 4, 6, 7]  # ignore hands
    for ignore_index in ignore_indexes:
        cross_indexes[ignore_index] = False
    assert sum(cross_indexes) > 0
    vis_bodies = ref_poses['bodies']['candidate'][cross_indexes]
    vis_ind = np.argsort(vis_bodies[:, 1])[-1]
    ref_coord = vis_bodies[vis_ind]
    vis_bodies = poses['bodies']['candidate'][cross_indexes]
    aln_coord = vis_bodies[vis_ind]
    # offset_w = int(ref_coord[0] - aln_coord[0])
    offset_w = 0
    offset_h = int(ref_coord[1] - aln_coord[1])
    return offset_w, offset_h


def _get_wh(points):
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    w = x_max - x_min
    h = y_max - y_min
    return w, h


def _get_face_indexes():
    left_face_indexes = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        18,
        19,
        20,
        21,
        22,
        32,
        33,
        37,
        38,
        39,
        40,
        41,
        42,
        49,
        50,
        51,
        59,
        60,
        61,
        62,
        68,
    ]
    mid_face_indexes = [9, 28, 29, 30, 31, 34, 52, 58, 63, 67]
    right_face_indexes = [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        23,
        24,
        25,
        26,
        27,
        35,
        36,
        43,
        44,
        45,
        46,
        47,
        48,
        53,
        54,
        55,
        56,
        57,
        64,
        65,
        66,
    ]
    face_indexes = left_face_indexes + mid_face_indexes + right_face_indexes
    assert len(list(set(face_indexes))) == len(face_indexes)
    assert len(left_face_indexes) == len(right_face_indexes)
    left_face_indexes = [idx - 1 for idx in left_face_indexes + mid_face_indexes]
    right_face_indexes = [idx - 1 for idx in right_face_indexes + mid_face_indexes]
    left_face_indexes.sort()
    right_face_indexes.sort()
    return left_face_indexes, right_face_indexes


def _draw_pose_images(poses_list, height, width, draw_body=True, draw_hand=False, draw_face=False):
    poses_list = copy.deepcopy(poses_list)
    _scale_offset_poses(poses_list, offset=(width // 2, height // 2))
    _scale_offset_poses(poses_list, scale=(1.0 / (width * 2), 1.0 / (height * 2)))
    _set_vis_indexes(poses_list)
    _clip_poses(poses_list)
    pose_images = []
    for poses in poses_list:
        pose_image = draw_poses(
            poses,
            height=height * 2,
            width=width * 2,
            draw_body=draw_body,
            draw_hand=draw_hand,
            draw_face=draw_face,
        )
        pose_image = pose_image.crop((width // 2, height // 2, width // 2 + width, height // 2 + height))
        pose_images.append(pose_image)
    return pose_images


def _get_num_vis(points):
    num_vis = 0
    for point in points:
        if _point_in_image(point):
            num_vis += 1
    return num_vis


def _check_face(poses, idx):
    faces = poses['faces'][idx]
    num_vis = _get_num_vis(faces)
    if num_vis != len(faces):
        return False
    left_point = faces[36]
    mid_point = faces[28]
    right_point = faces[45]
    if not (left_point[0] < mid_point[0] < right_point[0]):
        return False
    radio = (mid_point[0] - left_point[0]) / (right_point[0] - mid_point[0])
    if not (0.1 < radio < 10.0):
        return False
    return True


def _check_body(poses, idx):
    subset = poses['bodies']['subset']
    candidate = poses['bodies']['candidate'].reshape((subset.shape[0], -1, 2))
    subset = subset[idx]
    candidate = candidate[idx]
    for j in [0, 1, 2, 5, 8, 11]:
        if subset[j] < 0:
            return False
        if not _point_in_image(candidate[j]):
            return False
    # if subset[10] < 0 and subset[13] < 0 and not _point_in_image(candidate[10]) and not _point_in_image(candidate[13]):
    #     return False
    body_w = np.linalg.norm(candidate[2] - candidate[5])
    body_h_1 = np.linalg.norm(candidate[2] - candidate[8])
    body_h_2 = np.linalg.norm(candidate[5] - candidate[11])
    body_h = (body_h_1 + body_h_2) / 2
    radio = body_w / body_h
    if not (0.2 < radio < 2.0):
        return False
    return True
