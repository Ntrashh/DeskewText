import cv2
import numpy as np

from util import show


def rotate(img, angle):
    rows, cols = img.shape[:2]
    rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
    '''
    第一个参数：旋转中心点
    第二个参数：旋转角度
    第三个参数：缩放比例
    '''
    res = cv2.warpAffine(img, rotate, (cols, rows))
    return res


def rotate_image(template, angle, show=False):
    # 图片的中心点，这里假设是图片的中心
    center = (template.shape[1] // 2, template.shape[0] // 2)
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转图片，注意需要计算旋转后的新边界框以避免内容裁剪
    ppp = cv2.warpAffine(template, rotation_matrix, (template.shape[1], template.shape[0]), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)

    if show:
        cv2.imshow('template', ppp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return ppp


def merge_contour_points(contours):
    """
    Collects all vertices of the minimum area rectangle for each contour.

    Args:
    contours (list): A list of contours to process.

    Returns:
    np.ndarray: An array of points from the minimum bounding rectangles of each contour.
    """
    all_points = []  # Use a list to collect all points

    for contour in contours:
        rect = cv2.minAreaRect(contour)  # Calculate the minimum area rectangle
        box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
        box = np.int32(box)  # Ensure the points are in integer format
        all_points.extend(box)  # Collect all points in the list

    # Convert list of points to a NumPy array
    merged_points = np.array(all_points)
    return merged_points


def draw_min_bounding_rect(img, contours):  # 图,轮廓点,返回包含所有轮廓点的最小外接矩形的图,矩形四角
    """
    Draw the minimum bounding rectangle for provided contours.
    """
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return img, box


def getAngle(box):
    """
    Calculate the angle of the line with respect to the horizontal axis.

    Args:
    box (list): A list containing two points (x1, y1) and (x2, y2) which form a line.

    Returns:
    float: The angle in degrees between the line and the horizontal axis, adjusted to range [-45, 45].
    """
    x1, y1 = box[0]
    x2, y2 = box[1]

    # 计算角度
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    # 角度调整到 [-45, 45] 范围
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    return angle


def filter_contours_near_boundary(img, contours, boundary_thresh=1):
    """
    过滤掉靠近图像边缘的轮廓点。

    参数:
    img (numpy.ndarray): 输入图像，用于获取图像尺寸。
    contours (list): 轮廓列表，每个轮廓是一个点的集合。
    boundary_thresh (int): 边界阈值，决定多靠近边缘的点应该被过滤。

    返回:
    list: 过滤后保留的轮廓点。
    """
    filtered_contours = []
    img_height, img_width = img.shape[:2]

    for contour in contours:
        for point in contour:
            x, y = point[0]  # 点的坐标
            if not (x < boundary_thresh or x > img_width - boundary_thresh or
                    y < boundary_thresh or y > img_height - boundary_thresh):
                # 如果点不在边界阈值内，保留该点
                filtered_contours.append(point)

    return filtered_contours


def auto_correct_image(img):  # 传入单通道灰度图
    if len(img.shape) == 3:  # 彩色图有三个通道
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    # 确保图像类型为 uint8
    if gray_image.dtype != np.uint8:
        gray_image = np.uint8(gray_image)
    # 应用二值化
    ret, img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤掉靠近边界的轮廓
    filtered_contours = filter_contours_near_boundary(img, contours, 1)
    # 绘制轮廓
    contours = merge_contour_points(filtered_contours)  # 合并轮廓点
    img, box = draw_min_bounding_rect(img, contours)  # 画最小外接矩形
    angle = getAngle(box)
    img = rotate_image(img, angle)
    return img
