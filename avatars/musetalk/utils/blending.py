from PIL import Image
import numpy as np
import cv2
import copy
import os


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _odd_int(value, minimum=3):
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _canonical_mouth_mask(ori_shape, face_box, crop_box):
    """Build one stable, crop-relative mouth mask instead of per-frame parsing.

    The mask is authored in 256-space, softly feathered there, then resized to
    the actual crop. That keeps the transition width stable and prevents a
    4K crop from getting a very wide alpha halo.
    """
    crop_w, crop_h = ori_shape
    x, y, x1, y1 = [float(v) for v in face_box]
    x_s, y_s, _, _ = [float(v) for v in crop_box]
    rel_x0 = (x - x_s) / max(1.0, crop_w) * 256.0
    rel_x1 = (x1 - x_s) / max(1.0, crop_w) * 256.0
    rel_y0 = (y - y_s) / max(1.0, crop_h) * 256.0
    rel_y1 = (y1 - y_s) / max(1.0, crop_h) * 256.0

    face_w = max(1.0, rel_x1 - rel_x0)
    face_h = max(1.0, rel_y1 - rel_y0)
    center_x = int(round((rel_x0 + rel_x1) * 0.5))
    center_y = int(round(rel_y0 + face_h * float(os.getenv("LIVETALKING_CANONICAL_MOUTH_CENTER_Y", "0.68"))))
    axis_x = int(round(face_w * float(os.getenv("LIVETALKING_CANONICAL_MOUTH_AXIS_X", "0.34"))))
    axis_y = int(round(face_h * float(os.getenv("LIVETALKING_CANONICAL_MOUTH_AXIS_Y", "0.16"))))
    axis_x = max(8, min(110, axis_x))
    axis_y = max(5, min(70, axis_y))

    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask_256, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)

    kernel = _odd_int(os.getenv("LIVETALKING_MOUTH_FEATHER_KERNEL_256", "13"))
    sigma = max(0.0, float(os.getenv("LIVETALKING_MOUTH_FEATHER_SIGMA", "4.0")))
    mask_256 = cv2.GaussianBlur(mask_256, (kernel, kernel), sigma)
    return cv2.resize(mask_256, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    if mode in ("mouth", "jaw") and _env_bool("LIVETALKING_CANONICAL_MOUTH_MASK", False):
        mask_array = _canonical_mouth_mask(ori_shape, face_box, crop_box)
    else:
        mask_image = face_seg(face_large, mode=mode, fp=fp)
        mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
        mask_image = Image.new('L', ori_shape, 0)
        mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

        # keep upper_boundary_ratio of talking area
        width, height = mask_image.size
        top_boundary = int(height * upper_boundary_ratio)
        modified_mask_image = Image.new('L', ori_shape, 0)
        modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

        if _env_bool("LIVETALKING_FIXED_MOUTH_FEATHER", False):
            base = cv2.resize(np.array(modified_mask_image), (256, 256), interpolation=cv2.INTER_LINEAR)
            kernel = _odd_int(os.getenv("LIVETALKING_MOUTH_FEATHER_KERNEL_256", "13"))
            sigma = max(0.0, float(os.getenv("LIVETALKING_MOUTH_FEATHER_SIGMA", "4.0")))
            base = cv2.GaussianBlur(base, (kernel, kernel), sigma)
            mask_array = cv2.resize(base, ori_shape, interpolation=cv2.INTER_LINEAR)
        else:
            blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
            mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
