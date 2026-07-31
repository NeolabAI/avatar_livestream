import numpy as np
import cv2

def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body[y_s:y_e, x_s:x_e].copy()
    dst_h, dst_w = face_large.shape[:2]

    # Place the generated face at the correct offset inside crop_box.
    # The original MuseTalk logic pastes face into crop_box at the face_box
    # offset (x-x_s, y-y_s). Pasting at (0,0) shifts the whole face region
    # and visibly distorts the avatar. We keep the .copy() optimisation
    # (C memcpy, much faster than copy.deepcopy) but restore the proper
    # offset and clamp to the destination bounds for safety.
    y_off = max(0, y - y_s)
    x_off = max(0, x - x_s)
    y_end = min(dst_h, y1 - y_s)
    x_end = min(dst_w, x1 - x_s)
    paste_h = max(0, y_end - y_off)
    paste_w = max(0, x_end - x_off)
    if paste_h > 0 and paste_w > 0:
        face_large[y_off:y_off + paste_h, x_off:x_off + paste_w] = face[:paste_h, :paste_w]

    mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image / 255).astype(np.float32)

    # Guard: mask and crop may differ by 1px after 1080p downscale of a 4K avatar.
    # blendLinear requires identical sizes, otherwise it raises and paste_back fails.
    if mask_image.shape[:2] != face_large.shape[:2]:
        mask_image = cv2.resize(mask_image, (face_large.shape[1], face_large.shape[0]))

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(
        face_large, body[y_s:y_e, x_s:x_e], mask_image, 1 - mask_image
    )

    return body
