import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IN_IMAGE_DIR = "in_img"
OUT_IMAGE_DIR = "out_img"
directory_path = Path(IN_IMAGE_DIR)
IN_IMAGE_ENTRIES = directory_path.iterdir()
out_directory_path = Path(OUT_IMAGE_DIR)
OUT_IMAGE_ENTRIES = out_directory_path.iterdir()

def show(img, name="logo"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_to_cv2(img_path):
    # 使用Pillow打开图像
    pil_image = Image.open(img_path)

    # 将Pillow图像转换成NumPy数组
    image_array = np.array(pil_image)

    # Pillow的数组是RGB顺序，转换成OpenCV的BGR顺序
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array_bgr


def cv2_to_byte(cv2_image):
    result, encoded_image = cv2.imencode('.jpg', cv2_image)
    # 将编码后的图像转换为字节流
    return encoded_image.tobytes()


def hash_md5(byte_data):
    hash_object = hashlib.md5(byte_data)
    return hash_object.hexdigest()


def image_contrast_std(image, background_image):
    return np.std(image-background_image)
