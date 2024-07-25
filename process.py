from pathlib import Path

import cv2
import numpy as np

from delta_image import background_removal
from ocr import Ocr
from text_rotate import auto_correct_image
from util import image_to_cv2, OUT_IMAGE_ENTRIES, image_contrast_std, show, cv2_to_byte

background_list = [image_to_cv2(entry) for entry in OUT_IMAGE_ENTRIES]


def rotate(image, angle):
    rows, cols = image.shape[:2]
    rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
    '''
    第一个参数：旋转中心点
    第二个参数：旋转角度
    第三个参数：缩放比例
    '''
    res = cv2.warpAffine(image, rotate, (cols, rows))
    return res


def read_all_background():
    return [image_to_cv2(entry) for entry in OUT_IMAGE_ENTRIES]


def find_background(image, background_list):
    min_std_score = float('inf')
    min_background = None
    for background in background_list:
        std_score = image_contrast_std(image, background)
        # 更新最小分数和对应的背景
        if std_score < min_std_score:
            min_std_score = std_score
            min_background = background
    return min_background


def process_clear_background(image, write_background=True):
    background = find_background(image, background_list)
    background_removal_image = background_removal(image, background)
    if write_background:
        return 255 - background_removal_image
    return background_removal_image


def process_rotate_image(image):
    image = image.copy()
    image_background_remove_byte = cv2_to_byte(image)
    points = Ocr().target_point(image_background_remove_byte)
    for point in points:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = point
        # 扩大裁剪框
        expand_by = 5
        x1_expanded = max(0, x1 - expand_by)
        y1_expanded = max(0, y1 - expand_by)
        x2_expanded = min(width, x2 + expand_by)
        y2_expanded = min(height, y2 + expand_by)
        cropped_image = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        cropped_image_later = auto_correct_image(cropped_image)
        image[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = cropped_image_later
    return image


def process(path):
    image = image_to_cv2(path)
    clear_background_image = process_clear_background(image)
    rotate_image = process_rotate_image(clear_background_image)
    combined_image = np.hstack((clear_background_image, rotate_image))
    show(combined_image)


if __name__ == '__main__':
    process("sample/14b676d9c5b89ac0e1c3728295f4d444_发-叔-位.png")

