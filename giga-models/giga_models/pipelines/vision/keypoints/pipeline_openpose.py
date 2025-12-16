import numpy as np
from controlnet_aux import OpenposeDetector as _OpenposeDetector
from controlnet_aux.open_pose import HWC3, BodyResult, Keypoint, PoseResult
from controlnet_aux.open_pose import draw_poses as _draw_poses
from controlnet_aux.open_pose import resize_image
from PIL import Image

from .... import utils
from ...pipeline import BasePipeline


class OpenPosePipeline(BasePipeline):
    def __init__(self, model_name):
        self.model = utils.wrap_call(OpenposeDetector.from_pretrained)(model_name)

    def to(self, device):
        self.device = device
        self.model.body_estimation.model.to(device)
        self.model.hand_estimation.model.to(device)
        self.model.face_estimation.model.to(device)
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


class OpenposeDetector(_OpenposeDetector):
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
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        H, W, _ = input_image.shape
        input_image = resize_image(input_image, detect_resolution)
        poses = self.detect_poses(input_image, include_hand, include_face)
        if return_image:
            detected_map = _draw_poses(
                poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face
            )
            detected_map = Image.fromarray(detected_map)
            return detected_map
        else:
            poses = convert_poses(poses)
            return poses


def convert_poses(poses):
    new_poses = []
    for pose in poses:
        body = None
        left_hand = None
        right_hand = None
        face = None
        if pose.body is not None:
            body = {
                'total_score': pose.body.total_score,
                'total_parts': pose.body.total_parts,
                'keypoints': [
                    {'x': kp.x, 'y': kp.y, 'score': kp.score, 'id': kp.id} if kp is not None else None
                    for kp in pose.body.keypoints
                ],
            }
        if pose.left_hand is not None:
            left_hand = [
                {'x': kp.x, 'y': kp.y, 'score': kp.score, 'id': kp.id} if kp is not None else None
                for kp in pose.left_hand
            ]
        if pose.right_hand is not None:
            right_hand = [
                {'x': kp.x, 'y': kp.y, 'score': kp.score, 'id': kp.id} if kp is not None else None
                for kp in pose.right_hand
            ]
        if pose.face is not None:
            face = [
                {'x': kp.x, 'y': kp.y, 'score': kp.score, 'id': kp.id} if kp is not None else None for kp in pose.face
            ]
        new_poses.append({'body': body, 'left_hand': left_hand, 'right_hand': right_hand, 'face': face})
    return new_poses


def draw_poses(poses, height, width, draw_body=True, draw_hand=False, draw_face=False):
    new_poses = []
    for pose in poses:
        body = pose['body']
        left_hand = pose['left_hand']
        right_hand = pose['right_hand']
        face = pose['face']
        new_poses.append(
            PoseResult(
                BodyResult(
                    keypoints=[
                        Keypoint(x=keypoint['x'], y=keypoint['y']) if keypoint is not None else None
                        for keypoint in body['keypoints']
                    ],
                    total_score=body['total_score'],
                    total_parts=body['total_parts'],
                ),
                [Keypoint(x=keypoint['x'], y=keypoint['y']) if keypoint is not None else None for keypoint in left_hand]
                if left_hand is not None
                else None,
                [
                    Keypoint(x=keypoint['x'], y=keypoint['y']) if keypoint is not None else None
                    for keypoint in right_hand
                ]
                if right_hand is not None
                else None,
                [Keypoint(x=keypoint['x'], y=keypoint['y']) if keypoint is not None else None for keypoint in face]
                if face is not None
                else None,
            )
        )
    image = _draw_poses(new_poses, height, width, draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face)
    image = Image.fromarray(image)
    return image
