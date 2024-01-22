import cv2
from PIL import Image, ImageOps
from face_matcher.logger import logger
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
from scipy.ndimage import binary_erosion


class HumanCrop(object):
    face_seg_model_path = "models/yolo/selfie_multiclass_256x256.tflite"
    pose_model_path = "models/yolo/pose_landmarker_heavy.task"
    face_model_path = "models/yolo/blaze_face_short_range.tflite"

    def get_face_seg(self, image: np.ndarray, padding_pixel: int = 0, mask_invert: bool = False):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # Create a image segmenter instance with the image mode:
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.face_seg_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True)
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            segmented_masks = segmenter.segment(mp_image)
            category_mask = segmented_masks.category_mask
            category_hair = 1
            category_face = 3
            BG_COLOR = (0, 0, 0)  # black
            MASK_COLOR = (255, 255, 255)  # white
            # Generate solid color images for showing the output segmentation mask.
            image_data = mp_image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            stack = np.stack((category_mask.numpy_view(),) * 3, axis=-1)
            condition_face = stack == category_face
            condition_hair = stack == category_hair
            output_image = np.where(np.logical_or(condition_face, condition_hair), fg_image, bg_image)
            output_image_pil = Image.fromarray(output_image).convert('L')
            output_image_binary_mask = np.asarray(output_image_pil) > 0
            output_image = binary_erosion(output_image_binary_mask, iterations=padding_pixel)
            output_image = Image.fromarray(output_image.astype(np.uint8) * 255, mode='L').convert('RGB')
            if mask_invert:
                output_image = ImageOps.invert(output_image)
        return np.asarray(output_image)

    def get_face_crop(self, image: np.ndarray, resize_width: int):
        options = vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.face_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        width = image.shape[1]
        height = image.shape[0]
        logger.info(f"Image width : {width}, height : {height}")
        with vision.FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)
            if len(result.detections) > 1:
                logger.info("More then 1 face")
            if len(result.detections) == 0:
                logger.info("No face")
            detection = result.detections[0]
            origin_bbox = detection.bounding_box
            logger.info(f"origin_bbox : {origin_bbox}")
            bbox_width = min(origin_bbox.width * 3, width)
            diff = (bbox_width - origin_bbox.width) // 2
            bbox_left = max(origin_bbox.origin_x - diff, 0)
            bbox_top = max(origin_bbox.origin_y - diff, 0)
            bbox = (
                bbox_left,
                bbox_top,
                bbox_left + bbox_width,
                bbox_top + bbox_width
            )
            logger.info(f"bbox : {bbox}")
        image_pil = Image.fromarray(image)
        image_pil = image_pil.crop(box=bbox)
        wpercent = (resize_width / float(image_pil.size[0]))
        hsize = int((float(image_pil.size[1]) * float(wpercent)))
        image_pil = image_pil.resize((resize_width, hsize), Image.Resampling.LANCZOS)
        return np.asarray(image_pil)

    def get_should_pose(self, image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.75
        )
        with vision.PoseLandmarker.create_from_options(options) as poser:
            pose_estimate_result = poser.detect(mp_image)
            if len(pose_estimate_result.pose_landmarks) > 1:
                logger.error("Multiple human detected")
            elif len(pose_estimate_result.pose_landmarks) == 0:
                logger.error("No human detected")

            # Pose landmark detail
            # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
            left_shoulder_landmark = pose_estimate_result.pose_landmarks[0][11]  # 人物左手肩膀/画面右边肩膀
            right_shoulder_landmark = pose_estimate_result.pose_landmarks[0][12]  # 人物右手肩膀/画面左边肩膀
            logger.info(
                f"left_shoulder_landmark: {left_shoulder_landmark}, right_shoulder_landmark: {right_shoulder_landmark}")
            shoulder_width = int(abs(right_shoulder_landmark.x - left_shoulder_landmark.x))
            image_pil = Image.fromarray(image)
            bbox = (int(min(right_shoulder_landmark.x, left_shoulder_landmark.x)),
                    0, shoulder_width,
                    int(max(right_shoulder_landmark.y, left_shoulder_landmark.y)))
            logger.info(f"bbox to shoulder : {bbox}")
        image_pil = image_pil.crop(box=bbox)
        return np.asarray(image_pil)
