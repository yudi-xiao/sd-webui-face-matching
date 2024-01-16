import cv2
from collections import namedtuple
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops


def use_gpu_opencv():
    return True


def inference_bbox(
        model,
        image: Image.Image,
        confidence: float = 0.3,
        device: str = "",
):
    pred = model(image, conf=confidence, device=device, verbose=False)
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    else:
        # Handle the grayscale image here
        # For example, you might want to convert it to a 3-channel grayscale image for consistency:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        kernel = cv2.UMat(kernel)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        if use_gpu_opencv():
            cv2_mask = cv2.UMat(cv2_mask)

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iter)

        if use_gpu_opencv():
            dilated_mask = dilated_mask.get()

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


def crop_image(image, crop_region):
    return crop_ndarray4(image, crop_region)


def load_image_mask_tensor(image_path: Image) -> tuple[torch.Tensor, torch.Tensor]:
    i = image_path
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))


def filter(segs, labels):
    labels = set([label.strip() for label in labels])

    if 'all' in labels:
        return (segs, (segs[0], []),)
    else:
        res_segs = []
        remained_segs = []

        for x in segs[1]:
            if x.label in labels:
                res_segs.append(x)
            elif 'eyes' in labels and x.label in ['left_eye', 'right_eye']:
                res_segs.append(x)
            elif 'eyebrows' in labels and x.label in ['left_eyebrow', 'right_eyebrow']:
                res_segs.append(x)
            elif 'pupils' in labels and x.label in ['left_pupil', 'right_pupil']:
                res_segs.append(x)
            else:
                remained_segs.append(x)

    return ((segs[0], res_segs), (segs[0], remained_segs),)


def read_crop_data(segs):
    crop = []
    img = []

    h = segs[0][0]
    w = segs[0][1]

    mask = np.zeros((h, w), dtype=np.uint8)
    if len(segs[1]) < 1:
        raise Exception("No human detected!")

    crop.append(segs[1][0].crop_region)
    img.append(segs[1][0].cropped_image)
    cropped_mask = segs[1][0].cropped_mask
    crop_region = segs[1][0].crop_region
    mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

    crop_img = torch.Tensor(img[0][0])[None,]
    mask_img = torch.from_numpy(mask.astype(np.float32) / 255.0)

    left, top, right, bottom = crop[0]
    width = right - left
    height = bottom - top

    return (segs, crop_img, ((width, height), crop[0]), mask_img,)


def creatEmptyImage(image, gray_value=0.5):
    width = image.shape[2]
    height = image.shape[1]

    white = np.ones((height, width, 3), np.float32)
    white = torch.from_numpy(white)[None,]

    black = np.zeros((height, width, 3), np.float32)
    black = torch.from_numpy(black)[None,]

    gray = np.zeros((height, width, 3), np.float32)
    gray.fill(gray_value)
    gray = torch.from_numpy(gray)[None,]
    return (black, white, gray, width, height,)


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def apply_overlay_image(base_image, overlay_image, optional_mask=None):
    mask = optional_mask
    overlay_image = tensor2pil(overlay_image)

    # Add Alpha channel to overlay
    overlay_image = overlay_image.convert('RGBA')
    overlay_image.putalpha(Image.new("L", overlay_image.size, 255))

    # If mask connected, check if the overlay_image image has an alpha channel
    if mask is not None:
        # Convert mask to pil and resize
        mask = tensor2pil(mask)
        mask = mask.resize(overlay_image.size)
        # Apply mask as overlay's alpha
        overlay_image.putalpha(ImageOps.invert(mask))

    # Split the base_image tensor along the first dimension to get a list of tensors
    base_image_list = torch.unbind(base_image, dim=0)

    # Convert each tensor to a PIL image, apply the overlay, and then convert it back to a tensor
    processed_base_image_list = []
    for tensor in base_image_list:
        # Convert tensor to PIL Image
        image = tensor2pil(tensor)

        # Paste the overlay image onto the base image
        if mask is None:
            image.paste(overlay_image, (0, 0))
        else:
            image.paste(overlay_image, (0, 0), overlay_image)

        # Convert PIL Image back to tensor
        processed_tensor = pil2tensor(image)

        # Append to list
        processed_base_image_list.append(processed_tensor)

    # Combine the processed images back into a single tensor
    base_image = torch.stack([tensor.squeeze() for tensor in processed_base_image_list])

    # Return the edited base image
    return base_image


def paste_image(image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1):
    def lingrad(size, direction, white_ratio):
        image = Image.new('RGB', size)
        draw = ImageDraw.Draw(image)
        if direction == 'vertical':
            black_end = int(size[1] * (1 - white_ratio))
            range_start = 0
            range_end = size[1]
            range_step = 1
            for y in range(range_start, range_end, range_step):
                color_ratio = y / size[1]
                if y <= black_end:
                    color = (0, 0, 0)
                else:
                    color_value = int(((y - black_end) / (size[1] - black_end)) * 255)
                    color = (color_value, color_value, color_value)
                draw.line([(0, y), (size[0], y)], fill=color)
        elif direction == 'horizontal':
            black_end = int(size[0] * (1 - white_ratio))
            range_start = 0
            range_end = size[0]
            range_step = 1
            for x in range(range_start, range_end, range_step):
                color_ratio = x / size[0]
                if x <= black_end:
                    color = (0, 0, 0)
                else:
                    color_value = int(((x - black_end) / (size[0] - black_end)) * 255)
                    color = (color_value, color_value, color_value)
                draw.line([(x, 0), (x, size[1])], fill=color)

        return image.convert("L")

    crop_size, (left, top, right, bottom) = crop_data
    crop_image = crop_image.resize(crop_size)

    if sharpen_amount > 0:
        for _ in range(int(sharpen_amount)):
            crop_image = crop_image.filter(ImageFilter.SHARPEN)

    blended_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
    blended_mask = Image.new('L', image.size, 0)
    crop_padded = Image.new('RGBA', image.size, (0, 0, 0, 0))
    blended_image.paste(image, (0, 0))
    crop_padded.paste(crop_image, (left, top))
    crop_mask = Image.new('L', crop_image.size, 0)

    if top > 0:
        gradient_image = ImageOps.flip(lingrad(crop_image.size, 'vertical', blend_amount))
        crop_mask = ImageChops.screen(crop_mask, gradient_image)

    if left > 0:
        gradient_image = ImageOps.mirror(lingrad(crop_image.size, 'horizontal', blend_amount))
        crop_mask = ImageChops.screen(crop_mask, gradient_image)

    if right < image.width:
        gradient_image = lingrad(crop_image.size, 'horizontal', blend_amount)
        crop_mask = ImageChops.screen(crop_mask, gradient_image)

    if bottom < image.height:
        gradient_image = lingrad(crop_image.size, 'vertical', blend_amount)
        crop_mask = ImageChops.screen(crop_mask, gradient_image)

    crop_mask = ImageOps.invert(crop_mask)
    blended_mask.paste(crop_mask, (left, top))
    blended_mask = blended_mask.convert("L")
    blended_image.paste(crop_padded, (0, 0), blended_mask)

    return (blended_image.convert("RGB"), blended_mask.convert("RGB"))


def image_paste_crop(image, crop_image, crop_data=None, crop_blending=0.25, crop_sharpening=0):
    if crop_data == False:
        print("未找到crop_data!!!")
        return (image, pil2tensor(Image.new("RGB", tensor2pil(image).size, (0, 0, 0))))

    result_image, result_mask = paste_image(tensor2pil(image), tensor2pil(crop_image), crop_data,
                                            crop_blending, crop_sharpening)

    return (result_image, result_mask)


SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


class NO_SEGM_DETECTOR:
    pass


class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image: Image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # if isinstance(image, torch.Tensor):
        #     image = tensor2pil(image)
        detected_results = inference_bbox(self.bbox_model, image, threshold, device=device)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        # h = image.shape[1]
        h = image.height
        # w = image.shape[2]
        w = image.width

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(pil2tensor(image), crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = h, w
        return shape, items

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass
