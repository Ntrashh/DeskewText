"""
合成背景图
"""
import cv2
import numpy as np
from delta_image import background_removal
from ocr import Ocr
from text_rotate import merge_contour_points, draw_min_bounding_rect
from util import hash_md5, image_to_cv2, cv2_to_byte, IN_IMAGE_ENTRIES, OUT_IMAGE_DIR, show


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


def is_black_box(point, image):
    """
    判断指定坐标是否是黑色块
    :param point: 
    :param image:
    :return: 
    """
    x1, y1, x2, y2 = point
    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([0, 0, 0], dtype="uint8")
    region = image[y1:y2 + 1, x1:x2 + 1]
    # 创建黑色的掩膜
    black_mask = cv2.inRange(region, lower_black, upper_black)
    # 检查掩膜中是否有黑色像素
    if cv2.countNonZero(black_mask) > 0:
        return True
    return False


def full_image(point_lst, base_image, other_image):
    for point in point_lst:
        x1, y1, x2, y2 = point
        if is_black_box(point, other_image):
            continue
        else:
            base_image[y1:y2 + 1, x1:x2 + 1] = other_image[y1:y2 + 1, x1:x2 + 1]
    return base_image


def filter_background(background_list):
    min_diff_image = None
    min_num = 100
    for background in background_list:
        point_dict = dict()
        key = background["key"]
        image = background["image"]

        for _bg in background_list:
            if key == _bg["key"]:
                continue
            _image = _bg["image"]
            # 根据背景移除目标图片的背景显示差异
            background_removed_image = background_removal(_image, image)
            # show(background_removed_image)
            background_removed_image = background_removed_image.astype(np.uint8)
            # 获取差异后的轮廓点
            contours, _ = cv2.findContours(
                background_removed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = merge_contour_points(contour)  # 合并轮廓点
                _, box = draw_min_bounding_rect(background_removed_image, contour)
                k = "-".join([str(i) for i in box.flatten().tolist()])
                if point_dict.get(k) is None:
                    point_dict[k] = 1
                else:
                    point_dict[k] += 1
        max_key = max(point_dict, key=lambda k: point_dict[k])
        max_count = point_dict[max_key]
        if max_count >= len(background_list) - 1:
            continue
        if max_count < min_num:
            min_diff_image = image.copy()
            min_num = max_count
    return min_diff_image


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
            is_black = [is_black_box(point, copied_image) for point in point_lst]
            if not all(not item for item in is_black):
                continue

            image_bytes = cv2_to_byte(full_after_image)
            hash_key = hash_md5(image_bytes)

            if _fill_image_dict.get(hash_key) is None:
                _fill_image_dict[hash_key] = {"count": 0, "image": full_after_image}
            else:
                _fill_image_dict[hash_key]["count"] += 1
        if not _fill_image_dict:
            continue
        max_key = max(_fill_image_dict, key=lambda k: _fill_image_dict[k]["count"])
        max_image = _fill_image_dict[max_key]["image"]
        max_count = _fill_image_dict[max_key]["count"]
        # show(max_image, f"_{max_count}_{max_key}")
        if fill_image_dict.get(max_key) is None:
            fill_image_dict[max_key] = {"count": 0, "image": max_image}
        else:
            fill_image_dict[max_key]["count"] += 1
    max_key = max(fill_image_dict, key=lambda k: fill_image_dict[k]["count"])
    max_count = max(fill_image_dict[k]['count'] for k in fill_image_dict)
    background_list = [{"key": k, "image": v["image"]} for k, v in fill_image_dict.items() if v['count'] == max_count]
    if len(background_list) > 2:
        image = filter_background(background_list)
        if image:
            return image
    max_image = fill_image_dict[max_key]["image"]
    return max_image


def start():
    img_classification_dict = img_classification()
    for name, images in img_classification_dict.items():
        if len(images) < 8:
            print(f"{name} 底图太少可能造成分离不准确! 样本数:{len(images)}")
        image = splicing(images)
        cv2.imwrite(f'{OUT_IMAGE_DIR}/{name}.png', image)
        print(f"{name} 底图生成成功!")


if __name__ == '__main__':
    start()
