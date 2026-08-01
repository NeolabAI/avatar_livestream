import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet
import torchvision.transforms as transforms

class FaceParsing():
    def __init__(self, left_cheek_width=80, right_cheek_width=80):
        self.net = self.model_init()
        self.preprocess = self.image_preprocess()
        # Ensure all size parameters are integers
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height
        
        # Create kernel with explicit integer dimensions
        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2  # Ensure center coordinates are integers
        
        # Cone part
        for row in range(cone_height):
            if row < cone_height//2:
                continue
            width = int(2 * (row - cone_height//2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1

        # Vertical extension part
        if cone_height > 0:
            base_width = int(kernel[cone_height-1].sum())
        else:
            base_width = 1
        
        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width//2)))
            end = min(total_size, int(center_x + (base_width//2) + 1))
            kernel[row, start:end] = 1
        self.kernel = kernel
        
        # Modify cheek erosion kernel to be flatter ellipse
        self.cheek_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (35, 3))
        
        # Add cheek area mask (protect chin area)
        self.cheek_mask = self._create_cheek_mask(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
        
    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        """Create cheek area mask (1/4 area on both sides)"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)    # Left cheek
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)  # Right cheek
        return mask

    def model_init(self, 
                   resnet_path='./models/face-parse-bisent/resnet18-5c106cde.pth', 
                   model_pth='./models/face-parse-bisent/79999_iter.pth'):
        net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth)) 
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512), mode="raw"):
        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # Add 14:neck, remove 10:nose and 7:8:9
            if mode == "neck":
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[np.where(parsing!=255)] = 0
            elif mode == "mouth":
                mouth_region = (np.isin(parsing, [11, 12, 13]) * 255).astype(np.uint8)
                mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 19))
                mouth_region = cv2.dilate(mouth_region, mouth_kernel, iterations=1)
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
                mouth_region = cv2.morphologyEx(mouth_region, cv2.MORPH_CLOSE, close_kernel)
                parsing[:, :] = 0
                parsing[mouth_region == 255] = 255
            elif mode == "mouth_chin":
                mouth_seed = (np.isin(parsing, [11, 12, 13]) * 255).astype(np.uint8)
                mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 17))
                mouth_core = cv2.dilate(mouth_seed, mouth_kernel, iterations=1)
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
                mouth_core = cv2.morphologyEx(mouth_core, cv2.MORPH_CLOSE, close_kernel)

                alpha = np.zeros_like(mouth_core, dtype=np.uint8)
                if np.count_nonzero(mouth_core) > 0:
                    ys, xs = np.where(mouth_core > 0)
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    mouth_w = max(12, x_max - x_min + 1)
                    mouth_h = max(8, y_max - y_min + 1)
                    cx = int(round((x_min + x_max) * 0.5))

                    perioral = cv2.dilate(
                        mouth_core,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 25)),
                        iterations=1,
                    )
                    alpha[perioral > 0] = np.maximum(alpha[perioral > 0], 190)

                    chin = np.zeros_like(alpha, dtype=np.uint8)
                    chin_center_y = int(round(y_max + mouth_h * 0.95))
                    chin_axis_x = int(round(mouth_w * 0.62))
                    chin_axis_y = int(round(mouth_h * 1.15))
                    cv2.ellipse(
                        chin,
                        (cx, min(511, chin_center_y)),
                        (max(18, chin_axis_x), max(12, chin_axis_y)),
                        0,
                        0,
                        360,
                        140,
                        -1,
                    )
                    skin = (np.isin(parsing, [1]) * 255).astype(np.uint8)
                    chin = cv2.bitwise_and(chin, skin)
                    alpha = np.maximum(alpha, chin)
                    alpha[mouth_core > 0] = 255

                parsing[:, :] = alpha
            elif mode == "jaw":
                face_region = np.isin(parsing, [1])*255
                face_region = face_region.astype(np.uint8)
                original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
                face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
                parsing[(face_region==255) & (~np.isin(parsing, [10]))] = 255         
                parsing[np.isin(parsing, [11, 12, 13])] = 255
                parsing[np.where(parsing!=255)] = 0
            else:
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[np.where(parsing!=255)] = 0

        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing

if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')
    
