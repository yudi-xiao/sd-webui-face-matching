import os.path

import torch
from PIL import Image
import modules.scripts as scripts
import gradio as gr
import logging
import sys
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_matcher.core import UltraBBoxDetector, paste_image, read_crop_data, filter, tensor2pil, pil2tensor
from modules.processing import StableDiffusionProcessingImg2Img
import numpy as np

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

logger = logging.getLogger('FaceMatcher')
logger.propagate = False
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)




class FaceMatcher(scripts.Script):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Face matcher")

    def title(self):
        return "Face Matcher"

    def show(self, is_img2img):
        if is_img2img:
            return scripts.AlwaysVisible
        else:
            return False

    def ui(self, is_img2img):
        with gr.Accordion("Face Matcher", open=False):
            with gr.Group():
                with gr.Row():
                    enable_checkbox = gr.Checkbox(False, label="Enable")
                    mode = gr.Dropdown(label="Mode", choices=["match_face", "face_seg"])
                with gr.Row():
                    head_scale = gr.Slider(minimum=1.0, maximum=2.0, value=1.4, label="Head Scale")
                with gr.Row():
                    crop_blending = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Crop Blending")
                with gr.Row():
                    crop_sharpening = gr.Slider(minimum=-1.0, maximum=1.0, value=0, label="Crop Sharpening")
                with gr.Row():
                    location_image = gr.Image(source="upload", type="numpy", image_mode="RGBA", label="Face")
        return [enable_checkbox, mode, head_scale, location_image, crop_blending, crop_sharpening]

    def before_process(self, p: StableDiffusionProcessingImg2Img, *args, **kwargs):
        enable = args[0]
        if enable:
            mode = args[1]
            if mode == "match_face":
                # need to use --disable-safe-unpickle when start up webui, ref:
                # https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2235
                model_path = "models/yolo/face_yolov8m.pt"
                if os.path.exists(model_path):
                    try:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        yolo_model = YOLO(model_path)
                        bbox_detector = UltraBBoxDetector(yolo_model)
                    except Exception as e:
                        logger.exception(e)
                else:
                    logger.error(f"Model not exist at {model_path}")
                head_scale = args[2]
                face_image = args[3]
                face_image = Image.fromarray(face_image)
                crop_blending = args[4]
                crop_sharpening = args[5]
                sd_imgs = []
                avatar_segs = bbox_detector.detect(face_image, threshold=0.50, dilation=10, crop_factor=head_scale,
                                                   drop_size=5, detailer_hook=None)
                for init_image in p.init_images:
                    location_segs = bbox_detector.detect(init_image, threshold=0.50, dilation=10,
                                                         crop_factor=head_scale, drop_size=5, detailer_hook=None)
                    try:
                        _, avatar_image_segs, _, _ = read_crop_data(avatar_segs)
                    except Exception as e:
                        logger.exception(e)
                    try:
                        _, _, location_image_seg_data, _ = read_crop_data(location_segs)
                    except Exception as e:
                        logger.exception(e)
                    out_img, _ = paste_image(init_image, tensor2pil(avatar_image_segs), location_image_seg_data,
                                             crop_blending, crop_sharpening)
                    out = "temp/output.png"
                    if not os.path.exists(os.path.dirname(out)):
                        os.makedirs(os.path.dirname(out))
                    out_img.save(out)
                    sd_imgs.append(out_img)
                p.init_images = sd_imgs
            elif mode == "face_seg":
                for init_image in p.init_images:
                    # todo : handle face seg
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(init_image))
                    model_path = "models/yolo/selfie_multiclass_256x256.tflite"
                    # Create a image segmenter instance with the image mode:
                    options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path),
                                                    running_mode=VisionRunningMode.IMAGE,
                                                    output_category_mask=True)
                    with ImageSegmenter.create_from_options(options) as segmenter:
                        segmented_masks = segmenter.segment(mp_image)
                        category_mask = segmented_masks.category_mask
                        category_face = 3
                        BG_COLOR = (0, 0, 0)  # black
                        MASK_COLOR = (255, 255, 255)  # white

                        # Generate solid color images for showing the output segmentation mask.
                        image_data = mp_image.numpy_view()
                        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
                        fg_image[:] = MASK_COLOR
                        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
                        bg_image[:] = BG_COLOR
                        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == category_face
                        output_image = np.where(condition, fg_image, bg_image)
                        output_image = Image.fromarray(output_image)
                    p.image_mask = output_image
