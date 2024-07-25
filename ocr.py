import base64
import io
import re
from collections import Counter
from typing import List
import cv2
import ddddocr
import numpy as np
from PIL import Image


class Ocr:
    def __init__(self):
        self.target_ocr = ddddocr.DdddOcr(det=True, show_ad=False)
        self.ocr = ddddocr.DdddOcr(show_ad=False,beta=True)

    def recognize_text(self, image_byte) -> str:
        """
        识别文字
        :param image_byte:
        :return:
        """
        text = self.ocr.classification(image_byte)
        if text is None:
            return ""
        return text

    def crop_image(self, image: Image, coordinate: List[int]) -> bytes:
        """
        剪切文字，并将Image转换为字节流
        :param image:
        :param coordinate:
        :return:
        """
        cropped_image = image.crop(coordinate)
        output = io.BytesIO()
        cropped_image.save(output, format='PNG')
        image_bytes = output.getvalue()
        return image_bytes

    def target_recognition(self, image_byte) -> List[List[int]]:
        """
        识别目标
        :return:
        """
        bboxes = self.target_ocr.detection(image_byte)
        return bboxes

    def target_point(self, image_byte):
        """
        识别目标的位置
        :param image_byte:
        :return:
        """
        return self.target_recognition(image_byte)

    def text_point(self, image_byte):
        """
        识别文字的位置
        :param image_byte:
        :return:
        """
        text_point = {}
        bboxes = self.target_ocr.detection(image_byte)
        image_stream = io.BytesIO(image_byte)
        image = Image.open(image_stream)
        for bbox in bboxes:
            cropped_image = self.crop_image(image, bbox)
            text = self.recognize_text(cropped_image)
            text_point[text] = bbox
        return text_point
