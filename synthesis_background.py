"""
合成背景图
"""
import cv2
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from ocr import Ocr

IN_IMAGE_DIR = "in_img"
OUT_IMAGE_DIR = "out_img"
directory_path = Path(IN_IMAGE_DIR)
IN_IMAGE_ENTRIES = directory_path.iterdir()


def show(img, name="logo"):
    cv2.imshow(name,img)
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


def cv2_to_byte(cv2_imgae):
    result, encoded_image = cv2.imencode('.jpg', cv2_imgae)
    # 将编码后的图像转换为字节流
    return encoded_image.tobytes()


def hash_md5(byte_data):
    hash_object = hashlib.md5(byte_data)
    return hash_object.hexdigest()


def hash_background(img):
    height, width, channels = img.shape
    # 提取四个角的像素值
    top_left = img[0, 0]
    top_right = img[0, width - 1]
    bottom_left = img[height - 1, 0]
    bottom_right = img[height - 1, width - 1]
    # 可以将这些像素值转换为字符串
    corner_data = f"{top_left}{top_right}{bottom_left}{bottom_right}"
    return hash_md5(corner_data.encode())


def img_classification():
    """
    对背景图进行分类
    :return:
    """
    img_classification_dict = {}
    for entry in IN_IMAGE_ENTRIES:
        image_array_bgr = image_to_cv2(entry)
        hash_name = hash_background(image_array_bgr)
        img_classification_dict.setdefault(hash_name, []).append(entry)
    return img_classification_dict


def img_set_point_black(image, point):
    x1, y1, x2, y2 = point
    # 将指定区域设置为黑色
    image[y1:y2 + 1, x1:x2 + 1] = (0, 0, 0)
    return image


def full_image(point_lst, base_image, other_image):
    for point in point_lst:
        x1, y1, x2, y2 = point

        lower_black = np.array([0, 0, 0], dtype="uint8")
        upper_black = np.array([0, 0, 0], dtype="uint8")
        region = other_image[y1:y2 + 1, x1:x2 + 1]
        # 创建黑色的掩膜
        black_mask = cv2.inRange(region, lower_black, upper_black)

        # 检查掩膜中是否有黑色像素
        if cv2.countNonZero(black_mask) > 0:
            continue
        else:
            base_image[y1:y2 + 1, x1:x2 + 1] = other_image[y1:y2 + 1, x1:x2 + 1]
    return base_image


def splicing(img_list):
    clear_text_images = {}
    for img_path in img_list:
        img_byte = open(img_path, 'rb').read()
        text_point = Ocr().text_point(img_byte)
        image_array = np.frombuffer(img_byte, np.uint8)
        # 解码图像数据
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        for text, point in text_point.items():
            if not text:
                continue
            image = img_set_point_black(image, point)
        img_key = hash_md5(img_byte)
        clear_text_images[img_key] = {
            "text_point": text_point,
            "image": image,
        }
    fill_image_dict = dict()
    for key, clear_text_image in clear_text_images.items():
        text_point = clear_text_image["text_point"]
        image = clear_text_image["image"]
        point_lst = text_point.values()
        copied_image = np.copy(image)
        _fill_image_dict = dict()
        for _key, _clear_text_image in clear_text_images.items():
            if _key == key:
                continue
            other_image = _clear_text_image["image"]
            copied_image = full_image(point_lst, copied_image, other_image)
            full_after_image = np.copy(copied_image)
            image_bytes = cv2_to_byte(full_after_image)
            hash_key = hash_md5(image_bytes)
            show(full_after_image,"full_after_image")
            show(other_image,"拼接")
            show(copied_image,"copied_image")
            if _fill_image_dict.get(hash_key) is None:
                _fill_image_dict[hash_key] = {"count": 0, "image": full_after_image}
            else:
                _fill_image_dict[hash_key]["count"] += 1

        max_key = max(_fill_image_dict, key=lambda k: _fill_image_dict[k]["count"])
        max_image = _fill_image_dict[max_key]["image"]
        max_count = _fill_image_dict[max_key]["count"]
        show(max_image, f"__{max_key}")
        if fill_image_dict.get(max_key) is None:
            fill_image_dict[max_key] = {"count": 0, "image": max_image}
        else:
            fill_image_dict[max_key]["count"] += 1

    max_key = max(fill_image_dict, key=lambda k: fill_image_dict[k]["count"])
    max_image = fill_image_dict[max_key]["image"]
    show(max_image, f"last_{max_key}")

    return max_image


def start():
    img_classification_dict = img_classification()
    for name, images in img_classification_dict.items():
        if name != "77dfe272d1758e542e7d6883b4d3e836":
            continue
        image = splicing(images)
        cv2.imwrite(f'{OUT_IMAGE_DIR}/{name}.png', image)


if __name__ == '__main__':
    start()
