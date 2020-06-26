import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

class plate_recognition:
    def __init__(self, img_path, wh_ratio_low, wh_ratio_high, Min_Area, kernel_size, blur):
        self.img_path = img_path
        self.wh_ratio_low = wh_ratio_low
        self.wh_ratio_high = wh_ratio_high
        self.Min_Area = Min_Area
        self.kernel_size = kernel_size
        self.blur = blur

    '''绘出掩码图片'''

    @staticmethod
    def draw(img_path, mask_blue, mask_green, mask_yellow):
        img1 = cv2.imread(img_path, 1)
        img2, img3, img4 = mask_blue, mask_green, mask_yellow

        cv2.namedWindow('img1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('img1', img1)

        cv2.namedWindow('blue', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('blue', img2)

        cv2.namedWindow('green', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('green', img3)

        cv2.namedWindow('yellow', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('yellow', img4)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    '''初始化实例'''

    @classmethod
    def initialize(cls, img_path, wh_ratio_low, wh_ratio_high, Min_Area, kernel_size, blur):
        assert bool(np.max(cv2.imread(img_path))), \
            "img can not be empty"
        assert wh_ratio_high > wh_ratio_low, \
            "ration_high must greater than ration_low"
        assert Min_Area >= 2000, \
            "min_area must greater than 2000"
        assert kernel_size > 1, \
            "kernel size must greater than 1"

        return cls(img_path, wh_ratio_low, wh_ratio_high, Min_Area, kernel_size, blur)

    '''
    图像处理第一步
    实现功能
    1：读取图片
    2：高斯去噪
    3：图像转hsv(由于hsv对黄色识别不敏感，且黄蓝色互为相反色，对黄色车牌做颜色反向处理）
    4：创建掩码
    5：开闭运算使得边缘相对平滑
    '''

    def plate_inRange(self):
        img_path = self.img_path
        blur = self.blur
        kernel_size = self.kernel_size

        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img, (blur, blur), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # find blue plate
        lower_blue = np.array([99, 34, 46])
        upper_blue = np.array([124, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        # find green plate
        lower_green = np.array([35, 34, 46])
        upper_green = np.array([99, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

        # find yellow plate
        # 黄色和蓝色是相反色
        img_reverse = 255 - img
        hsv_reverse = cv2.cvtColor(img_reverse, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([99, 34, 46])
        upper_blue = np.array([124, 255, 255])
        mask_yellow = cv2.inRange(hsv_reverse, lower_blue, upper_blue)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        return mask_blue, mask_green, mask_yellow

    '''
    图像处理第二步
    实现功能
    1：对上步骤返回的掩码图像查找contours
    2：筛选出contourArea大于2000的轮廓
    '''

    def findContours(self, mask_b, mask_g, mask_y):
        ##查找像素集合在Min_Area以上的contours
        Min_Area = self.Min_Area

        contours_b, hierarchy_b = cv2.findContours(mask_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_b = [cnt for cnt in contours_b if cv2.contourArea(cnt) > Min_Area]

        contours_g, hierarchy_g = cv2.findContours(mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_g = [cnt for cnt in contours_g if cv2.contourArea(cnt) > Min_Area]

        contours_y, hierarchy_y = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_y = [cnt for cnt in contours_y if cv2.contourArea(cnt) > Min_Area]

        return contours_b, contours_g, contours_y

    # 嵌套子函数
    # 实现功能：
    # 1：根据contour求取minAreaRect
    # 2：根据长宽比筛选出符合要求的minAreaRect
    # 3：过滤高大于宽的minAreaRect
    # 注：长宽比设定范围默认为2-5.5
    # 本代码默认车牌是横着的，且倾斜角度在30度以内
    def plate_box_selection(self, contours):
        wh_ratio_low = self.wh_ratio_low
        wh_ratio_high = self.wh_ratio_high
        car_contours = []
        car_contours_bak = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            bounding_w, bounding_h = cv2.boundingRect(cnt)[2:]
            area_width, area_height = rect[1]
            angle = rect[2]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            if area_height > 0 and area_width > 0:
                wh_ratio = area_width / area_height
                # 先假设车牌为普通车牌
                if wh_ratio > wh_ratio_low and wh_ratio < wh_ratio_high:
                    if 1.2 * bounding_h < bounding_w:
                        if not (30 < angle < 45 or -45 < angle < -30):
                            car_contours.append(rect)
        return car_contours

    '''
    图像处理第三步
    实现功能
    1：调用子函数plate_box_selection去除不符合长宽比，角度的contours
    '''

    def plate_boxes_selection(self, contours_b, contours_g, contours_y):
        ##对应找到的框和颜色  blue:1 green:2 yellow:3
        color_mark = []

        # 检测蓝色标准车牌和大车牌
        contours_b = self.plate_box_selection(contours_b)
        color_mark.extend([1] * len(contours_b))

        # 检测绿色标准车牌和大车牌
        contours_g = self.plate_box_selection(contours_g)
        color_mark.extend([2] * len(contours_g))

        # 检测黄色标准车牌和大车牌
        contours_y = self.plate_box_selection(contours_y)
        color_mark.extend([3] * len(contours_y))

        contours = contours_b + contours_g + contours_y

        if len(contours) > 0:
            return contours, color_mark
        else:
            return [], []

            # 嵌套子函数

    # 1：根据minAreaRect里的angle对倾斜车牌进行旋转
    def crop_rect(self, img, rect):
        # get the parameter of the small rectangle
        center, size, angle = rect[0], rect[1], rect[2]
        if (angle > -45):
            center, size = tuple([round(i) for i in center]), tuple([round(rect[1][0]), round(rect[1][1])])
            # angle-=270
            height, width = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D(center, angle, 1)
            img_rot = cv2.warpAffine(img, M, (width, height))
            img_crop = cv2.getRectSubPix(img_rot, size, center)
        else:
            center = tuple([round(i) for i in center])
            size = tuple([round(rect[1][1]), round(rect[1][0])])
            angle -= 270
            height, width = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D(center, angle, 1)
            img_rot = cv2.warpAffine(img, M, (width, height))
            img_crop = cv2.getRectSubPix(img_rot, size, center)
        return img_crop

    '''
    图像处理第四步
    实现功能
    1：调用子函数crop_rect对每个contour进行旋转操作
    '''

    def crop_recycle(self, contours):
        img = cv2.imread(self.img_path)
        img_crops = []
        for contour in contours:
            img_crop = self.crop_rect(img, contour)
            img_crops.append(img_crop)
        return img_crops

    '''
    图像处理第五步：
    实现代码
    1：将黄色图像做颜色反向处理,以便于后期做图像二值化处理
    '''

    def reverse_yellow_plate(self, img_crops, color_mark):
        for color in range(len(color_mark)):
            if color_mark[color] == 3:
                img_crops[color] = 255 - img_crops[color]
        return img_crops

    '''
    图像处理第六步：
    实现功能
    根据颜色比例筛选不合格contour
    '''

    def crops_filter_color(self, img_crops, color_mark, contours):
        crops = []
        color_mark_filter = []
        contours_filter = []
        for crop, color, contour in zip(img_crops, color_mark, contours):
            b, g, r = cv2.split(crop)
            b = np.mean(b)
            g = np.mean(g)
            r = np.mean(r)
            mean = (b + g + r) / 3
            # print(b/mean,g/mean)
            if (b / mean > 1.18) or (g / mean > 1.18):
                crops.append(crop)
                color_mark_filter.append(color)
                contours_filter.append(contour)
        return crops, color_mark_filter, contours_filter

    # 嵌套子函数
    # 实现功能：
    # 1：求取车牌高度方向的二值化图的像素累计
    # 2：横向从二值化图中间往两端移动，根据字符区和边缘处像素累计值较低，去除横向车牌边缘
    def crop_cut_ylim(self, crop):
        height, width = crop.shape[:2]
        y_hist = np.sum(crop, axis=1, dtype=np.float32) / 255
        threshold = np.mean(y_hist[height // 4:height * 3 // 4]) // 3
        peak = height // 2
        extra_rim = height // 20

        for climb in range(peak):
            i = peak - climb
            if y_hist[i] < threshold:
                if i - extra_rim > 0:
                    y_top = i - extra_rim
                    break
                else:
                    y_top = 0
            else:
                y_top = 0

        for climb in range(len(y_hist) - peak):
            i = peak + climb
            if y_hist[i] < threshold:
                if i + extra_rim < height:
                    y_bottom = i + extra_rim
                    break
                else:
                    y_bottom = height
            else:
                y_bottom = height

        crop = crop[y_top:y_bottom]
        return crop

    # 嵌套子函数
    # 实现功能
    # 去除车牌两端的黑框
    def check_black_rim(self, crop):
        height, width = crop.shape[:2]
        threshold_low = height // 10
        x_hist = np.sum(crop, axis=0, dtype=np.float32) / 255

        # check black border
        for left_border in range(width):
            if x_hist[left_border] <= threshold_low:
                continue
            else:
                break

        # check black border
        for right_border in range(width):
            if x_hist[width - right_border - 1] <= threshold_low:
                continue
            else:
                break

        right_border = width - right_border
        crop = crop[:, left_border:right_border]
        return crop

    # 嵌套子函数
    # 实现功能
    # 去除车牌两端的白框
    def check_white_rim(self, crop):
        height, width = crop.shape[:2]
        threshold_upper = 9 * height // 10
        x_hist = np.sum(crop, axis=0, dtype=np.float32) / 255

        # check white border
        for left_border in range(width):
            if x_hist[left_border] >= threshold_upper:
                continue
            else:
                break

        # check white border
        for right_border in range(width):
            if x_hist[width - right_border - 1] >= threshold_upper:
                continue
            else:
                break

        right_border = width - right_border
        crop = crop[:, left_border:right_border]
        return crop

    '''
    图片处理第七步：
    实现功能
    通过子函数crop_cut_ylim check_black_rim check_white_rim 对车牌边框进行切除
    '''

    def cut_rim(self, crops, color_mark, contours):
        if len(crops) > 0:
            license = []
            filter_contours = []
            for crop, color, contour in zip(crops, color_mark, contours):
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if color == 2:
                    ret, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    crop = 255 - crop
                else:
                    ret, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                crop = self.crop_cut_ylim(crop)
                crop = self.check_black_rim(crop)
                crop = self.check_white_rim(crop)
                crop = self.check_black_rim(crop)
                crop = self.check_white_rim(crop)
                crop = self.check_black_rim(crop)
                height, width = crop.shape[:2]
                if 5000 < height * width < 15000:
                    kernel = np.ones((2, 2), np.uint8)
                    crop = cv2.erode(crop, kernel)
                elif 15000 < height * width < 30000:
                    kernel = np.ones((3, 3), np.uint8)
                    crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
                elif height * width > 30000:
                    kernel = np.ones((5, 5), np.uint8)
                    crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
                license.append(crop)
                filter_contours.append(contour)
            return license, filter_contours
        else:
            return [], []

    '''
    图像处理第八步：
    实现功能
    车牌的二值化图每个字符之间是有间隔的，根据波谷去除不合格数据
    '''

    def crops_filter_by_vally(self, crops, color_mark, contours):
        crops_filter = []
        color_mark_filter = []
        contours_filter = []
        for crop, color, contour in zip(crops, color_mark, contours):
            height, width = crop.shape[:2]
            # waiting
            threshold_low = height // 8
            x_hist = np.sum(crop, axis=0, dtype=np.float32) / 255
            waves = np.array(x_hist > threshold_low, dtype=np.int32)
            vally = []
            for ii in range(1, len(waves) - 1):
                if waves[ii] == 0 and (waves[ii - 1] + waves[ii + 1]) == 1:
                    vally.append(ii)
            # print('vally'+str(len(vally)))
            if 2 < (len(vally) / 2) < 16:
                crops_filter.append(crop)
                color_mark_filter.append(color)
                contours_filter.append(contour)
        return crops_filter, color_mark_filter, contours_filter

    '''
    图像处理第九步：
    实现功能：
    容易受到树边框和蓝色路牌的影响，由于树和蓝色路牌一般在图片的上半部分，我们这里去掉图像上1/3的框
    '''

    def crops_filter_by_common(self, crops, color_mark, contours):
        img_path = self.img_path

        if len(crops) == 0:
            return [], [], []
        else:
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            if len(crops) == 1:
                return crops, color_mark, contours
            else:
                crops_filter = []
                color_mark_filter = []
                contours_filter = []
                for crop, color, contour in zip(crops, color_mark, contours):
                    y_coord = contour[0][1]
                    if y_coord > height // 3:
                        crops_filter.append(crop)
                        color_mark_filter.append(color)
                        contours_filter.append(contour)

            return crops_filter, color_mark_filter, contours_filter

    '''
    图像处理最后一步：
    实现功能：
    将剩下的框中面积最大的一个框作为车牌
    '''

    def crops_filter_size(self, crops, color_mark):
        areas = []
        if len(crops) < 1:
            return 0, 0
        else:
            for crop in crops:
                h, w = crop.shape[:2]
                area = h * w
                areas.append(area)
        return crops[np.argmax(areas)], color_mark[np.argmax(areas)]

    '''
    实现功能
    1. 高斯去噪
    2. 到add_weighted可以实现类似使用局部直方图均衡化 equalizeHist() createCLAHE可以达到的增加对比度的效果
    3. 图像二值化，查找边界
    4. 边界一体化
    '''

    def find_contours_by_Canny(self):
        img_path = self.img_path
        Min_Area = self.Min_Area

        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((20, 20), np.uint8)
        morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        img_opening = cv2.addWeighted(img, 1, morph, -1, 0)

        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_canny = cv2.Canny(img_thresh, 100, 200)

        kernel = np.ones((4, 19), np.uint8)
        img_edge1 = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)

        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]

        return contours

