import modules.scripts as scripts
import gradio as gr
from face_matcher.logger import logger
from face_matcher.human_crop import HumanCrop
from modules.processing import StableDiffusionProcessingImg2Img

class FaceMatcher(scripts.Script):
    def __init__(self):
        super().__init__()
        self.cropper = HumanCrop()
        logger.info("Initializing Face matcher")

    def title(self):
        return "Face Matcher"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def on_btn_clicked(self, image, mask_invert, padding):
        cropped_img = self.cropper.get_face_crop(image, resize_width=768)
        mask = self.cropper.get_face_seg(cropped_img, mask_invert=mask_invert, padding_pixel=padding)
        return cropped_img, mask

    def ui(self, is_img2img):
        with gr.Accordion("Face Matcher", open=False):
            with gr.Group():
                with gr.Row():
                    user_image = gr.Image(source="upload", type="numpy", image_mode="RGB", label="User Face")
                    crop_image = gr.Image(label="Cropped Image", type='numpy', image_mode="RGB")
                    crop_mask = gr.Image(label="Cropped Mask", type='numpy', image_mode="RGB")
                with gr.Row():
                    mask_invert = gr.Checkbox(label="Mask Invert")
                    padding = gr.Number(label="Padding", minimum=0, maximum=15, precision=0)
                with gr.Row():
                    btn = gr.Button(label="Gen")
                    btn.click(self.on_btn_clicked, inputs=[user_image, mask_invert, padding],
                              outputs=[crop_image, crop_mask])
        return [user_image]

    def before_process(self, p: StableDiffusionProcessingImg2Img, *args, **kwargs):
        pass
