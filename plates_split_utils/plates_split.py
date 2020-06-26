import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json


class plates_split:
    def __init__(self, plate_path, shape):
        self.plate_path = plate_path
        self.shape = shape

    '''初始化实例'''

    @classmethod
    def initialize(cls, plate_path, shape):
        assert np.max(cv2.imread(plate_path)), "plate image can not be empty"
        assert shape > 28, "plate shape must greater than 28"
        return cls(plate_path, shape)

    '''
    实现功能：
    1. canny查找边界
    2. 查找external轮廓
    3. 根据contour查找convexhull
    4. 过滤不符合位置和尺寸的convexhull
    5. 把剩下的hull按照x轴排序
    '''

    def find_external_convexHull(self):
        plate_path = self.plate_path
        img = cv2.imread(plate_path, 0)
        height, width = img.shape
        img = cv2.Canny(img, 100, 200)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        middle_range_low = height // 5
        middle_range_upper = height * 4 // 5
        min_area = height * width // 40
        max_area = height * width // 4.5

        rects = []
        # 计算每个轮廓点
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i], False)
            x, y, w, h = cv2.boundingRect(hull)
            if middle_range_low < (y + h / 2) < middle_range_upper:
                if min_area < w * h < max_area:
                    # print(x,y,w,h)
                    line = [x, y, w, h]
                    rects.append(line)

        rects = [rects[i] for i in list(np.argsort(np.array(rects)[:, 0]))]
        return rects

    '''
    实现功能：
    合并粘连框
    '''

    def filter_rects(self, rects):
        plate_path = self.plate_path
        c_remove_idx = []
        new_characters = []
        new_rects = []
        img = cv2.imread(plate_path, 0)
        height, width = img.shape
        w_mean = np.mean(np.array(rects)[:, 2])
        for i in range(len(rects) - 1):
            x, w = rects[i][0], rects[i][2]
            x_post, w_post = rects[i + 1][0], rects[i + 1][2]
            if (x + w - x_post) > 0:
                x_left = max(x, x_post)
                x_right = max(x + w, x_post + w_post)
                w_new = x_right - x_left
                if w_new < w_mean * 1.25:
                    c_remove_idx.append(i)
                    c_remove_idx.append(i + 1)

                    new_c = [x_left, 0, w_new, height]
                    new_characters.append(new_c)

        for i, rect in enumerate(rects):
            if i not in c_remove_idx:
                new_rects.append(rects[i])

        for new_c in new_characters:
            new_rects.append(new_c)

        new_rects = [new_rects[i] for i in list(np.argsort(np.array(new_rects)[:, 0]))]

        return new_rects

    '''
    实现功能：
    对多框少框做处理
    '''

    def adjust_rects(self, rects, color_mark):
        plate_path = self.plate_path
        img = cv2.imread(plate_path, 0)
        height, width = img.shape
        w_mean = np.mean(np.array(rects)[:, 2])  # 求字符平均长度
        w_sum = np.sum(np.array(rects)[:, 2])  # 求字符总长度
        rim = width = max(width - w_sum, 0) // 15  # 设置合适的字符边缘
        if color_mark == 2:  # 1:blue 2:green 3:yellow
            c_count = 8
        else:
            c_count = 7

        if len(rects) > c_count:  # 字符多了情况，实践证明，字符基本多在头尾处，进行检查补充
            rects_new = []
            for i, rect in enumerate(rects):
                if i in (0, len(rects) - 1):
                    if rect[2] > 3 * w_mean // 5:
                        rects_new.append(rect)
                else:
                    rects_new.append(rect)

        elif len(rects) < c_count:  # 字符少了情况，实践证明，字符基本少在头尾处，进行检查补充
            if rects[0][0] > w_mean:
                rect_chinese = [0, 0, int(rects[0][0] - 1), height]
                rects.insert(0, rect_chinese)
            elif width - (rects[len(rects) - 1][0] + rects[len(rects) - 1][2]) > w_mean:
                x_last = (rects[len(rects) - 1][0] + rects[len(rects) - 1][2]) + rim
                w_last = min(width - x_last, w_mean)
                rect_last = [x_last, 0, w_last, height]
                rects.insert(len(rects), rect_last)
            rects_new = rects

        else:
            rects_new = rects
        return rects_new

    '''
    实现功能：
    由于中文字符一般比别的字母要宽，而且中文有的有左右偏旁，有可能方框会从中间把中文分开，所以这里对中文字符进行宽度加长处理
    '''

    def adjust_chinese(self, rects):
        plate_path = self.plate_path
        img = cv2.imread(plate_path, 0)
        height, width = img.shape
        w_mean = np.mean(np.array(rects)[:, 2])
        w_sum = np.sum(np.array(rects)[:, 2])
        rim = width = max(width - w_sum, 0) // 15

        if rects[0][2] > w_mean * 1.1:
            return rects
        else:
            rects[0] = [max(int(rects[1][0] - w_mean * 1.15 - rim), 0), 0, int(w_mean * 1.15), height]
            return rects

    '''
    实现功能：
    字符分割
    字符填充边界
    '''

    def plate_split(self, rects):
        plate_path = self.plate_path
        shape = self.shape
        characters = []
        img = cv2.imread(plate_path, 0)
        for i, rect in enumerate(rects):
            left_rim = max(rect[0] - 1, 0)
            right_rim = rect[0] + rect[2] + 1
            character = img[:, left_rim:right_rim]
            height, width = character.shape
            border = max(height - width, 0) // 2
            character = cv2.copyMakeBorder(character, 0, 0, border, border, cv2.BORDER_CONSTANT, value=0)
            kernel = character.shape[0] // 30
            if kernel > 1:
                k = np.ones((kernel, kernel), np.uint8)
                character = cv2.morphologyEx(character, cv2.MORPH_CLOSE, k)
            character = cv2.resize(character, (shape, shape))
            # cv2.imshow(str(i),character)
            characters.append(character)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return characters
