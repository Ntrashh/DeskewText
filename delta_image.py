import cv2
from colormath import color_diff_matrix
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cmc, _get_lab_color1_vector, _get_lab_color2_matrix
import numpy as np



def load_image_and_convert_to_lab(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def _delta_e_cmc(color1, color2, pl=2, pc=1):
    color1_vector = _get_lab_color1_vector(color1)
    color2_matrix = _get_lab_color2_matrix(color2)
    delta_e = color_diff_matrix.delta_e_cmc(
        color1_vector, color2_matrix, pl=pl, pc=pc)[0]
    return delta_e.item()

def calculate_delta_e(img_lab, background_lab):
    height, width, _ = img_lab.shape
    delta_e_image = np.zeros((height, width))

    for j in range(height):
        for k in range(width):
            color1 = LabColor(lab_l=img_lab[j, k, 0], lab_a=img_lab[j, k, 1], lab_b=img_lab[j, k, 2])

            color2 = LabColor(lab_l=background_lab[j, k, 0], lab_a=background_lab[j, k, 1],
                              lab_b=background_lab[j, k, 2])
            delta_e_image[j, k] = _delta_e_cmc(color1, color2)
    return delta_e_image


def binary_threshold_delta_e(delta_e_image, threshold=10):
    # 应用阈值来二值化图像
    _, binary_image = cv2.threshold(delta_e_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def background_removal(img_lab, background_lab):
    # 计算 Delta E 图像
    delta_e_image = calculate_delta_e(img_lab, background_lab)

    binary_delta_e_image = binary_threshold_delta_e(delta_e_image)
    return binary_delta_e_image
