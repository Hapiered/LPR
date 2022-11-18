import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import temp
import recognition
import os

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
MIN_AREA = 500  # 车牌区域允许最大面积
PROVINCE_START = 1000


def point_limit(point, pic_width, pic_height):
    if point[0] < 0:
        point[0] = 0
    elif point[0] > pic_width:
        point[0] = pic_width
    if point[1] < 0:
        point[1] = 0
    elif point[1] > pic_height:
        point[1] = pic_height


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    # col_num_limit = self.cfg["col_num_limit"]
    row_num_limit = CFG["row_num_limit"]
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
# 

def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 根据找出的波峰，分隔图片，从而得到逐个字符图片


def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]: wave[1]])
    return part_cards


""" 预处理。获取已经矫正，或分区域后的车牌
    输入：图片彩色图或者图片地址
    输出：处理后的车牌，车牌颜色
"""


def pretreatment(filename, resize_rate=1):
    imgSrc = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    # 调整图片大小
    # 获取输入图像的高和宽
    pic_height, pic_width = imgSrc.shape[:2]
    img_copy = imgSrc.copy()
    if pic_width > MAX_WIDTH:
        change_rate = MAX_WIDTH / pic_width
        img_copy = cv2.resize(imgSrc, (MAX_WIDTH, int(pic_height * change_rate)), interpolation=cv2.INTER_AREA,)
        # 获取输入图像的高和宽
        pic_height, pic_width = img_copy.shape[:2]

    if resize_rate != 1:
        img_copy = cv2.resize(img_copy, (int(pic_width * resize_rate), int(pic_height * resize_rate)), interpolation=cv2.INTER_LANCZOS4)
        pic_height, pic_width = img_copy.shape[:2]

    oldimg = img_copy.copy()
    colors = [([15, 55, 55], [50, 255, 255]),  # 黄色
              ([100, 43, 46], [124, 255, 255]),  # 蓝色
              ([0, 3, 116], [76, 211, 255])]  # 绿色

    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, np.array(colors[0][0]), np.array(colors[0][1]))
    mask_blue = cv2.inRange(hsv, np.array(colors[1][0]), np.array(colors[1][1]))
    mask_green = cv2.inRange(hsv, np.array(colors[2][0]), np.array(colors[2][1]))
    # mask_blue = cv2.GaussianBlur(mask_blue,(5,5),0)
    # cv2.imshow("hsv", hsv)

    res_yellow = cv2.bitwise_and(hsv, hsv, mask=mask_yellow)
    res_blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    res_green = cv2.bitwise_and(hsv, hsv, mask=mask_green)
    # cv2.imshow("output1", res_blue)
    # 根据阈值找到对应颜色
    output = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    cv2.imshow("output2", output)

    # 1.灰度图的高斯平滑去噪
    gray_img_gauss = cv2.GaussianBlur(output, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    # cv2.imshow("output3_gauss", output)

    # # 找车牌轮廓
    # # 1.转灰度图，高斯平滑去噪
    # if CFG["blur"] > 0:
    #     img_gauss = cv2.GaussianBlur(img_copy, (CFG["blur"], CFG["blur"]), 0, 0, cv2.BORDER_DEFAULT)
    # oldimg = img_gauss
    # gray_img_gauss = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((5, 5), np.uint8)
    img_123 = cv2.morphologyEx(gray_img_gauss, cv2.MORPH_OPEN, kernel)
    cv2.imshow("img_123", img_123)
    cv2.imshow("gray_img_gauss0", gray_img_gauss)
    gray_img_gauss = cv2.addWeighted(gray_img_gauss, 1, img_123, -1, 0)
    cv2.imshow("gray_img_gauss1", gray_img_gauss)


    # 2.开运算，与上次运算结果融合
    # kernel = np.ones((3, 14), nuint8)
    # img_opening = cv2.morphologyEx(gray_img_gauss, cv2.MORPH_OPEN, kernel)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    # # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("img_opening1", img_opening)
    # kernel = np.ones((4, 19), np.uint8)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    # # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("img_opening2", img_opening)
    
    # cv2.imshow("img_opening1", img_opening)
    # img_opening = cv2.addWeighted(gray_img_gauss, 1, img_opening, -1, 0)
    # cv2.imshow("img_opening2", img_opening)
    # 3.二值化，找到图像边缘(不找边缘)
    ret, img_thresh = cv2.threshold(gray_img_gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("img_thresh", img_thresh)
    kernel = np.ones((3, 3), np.uint8)
    img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("img_opening1", img_opening)
    # kernel = np.ones((4, 19), np.uint8)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("img_opening2", img_opening)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("img_opening3", img_opening)
    
    img_edge = cv2.Canny(img_opening, 100, 200)
    # img_edge=img_thresh
    cv2.imshow("img_edge", img_edge)
    

    # 4.使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((5, 20), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # img_edge2=img_edge1
    cv2.imshow("img_edge1", img_edge1)
    cv2.imshow("img_edge2", img_edge2)
    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 6.逐个排除不是车牌的矩形区域，获取外接矩阵。（1）车牌最小面积，（2）车牌宽高比范围
    # 6.1（1）车牌最小面积
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    # 6.2（2）车牌宽高比范围
    car_contours = []
    oldimg1 = oldimg.copy()
    for cnt in contours:
        # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）
        rect = cv2.minAreaRect(cnt)
        rect_width, rect_height = rect[1]
        # 将长边设为宽，车牌有可能是竖直放置，矫正
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        wh_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if wh_ratio > 2 and wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            oldimg1 = cv2.drawContours(oldimg1, [box], 0, (0, 0, 255), 2)
            cv2.imshow("oldimg1", oldimg1)
    # 6.矫正矩形区域。矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    card_imgs = []
    for i, rect in enumerate(car_contours):
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 8, rect[1][1] + 8), angle)  # 扩大范围，避免车牌边缘被排除
        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_height]
        # 定位车牌最左，最右，最高，最低的点（车牌可能是不规则矩形）
        for point in box:
            if left_point[0] > point[0]:
                left_point = point.copy()
            if low_point[1] > point[1]:
                low_point = point.copy()
            if heigth_point[1] < point[1]:
                heigth_point = point.copy()
            if right_point[0] < point[0]:
                right_point = point.copy()
        # 左 低
        # 高 右
        if (left_point == heigth_point).all() or (left_point == low_point).all() \
                or (right_point == heigth_point).all() or (right_point == low_point).all():
            left_point[1] = low_point[1].copy()
            heigth_point[0] = left_point[0].copy()
            low_point[0] = right_point[0].copy()
            right_point[1] = heigth_point[1].copy()

        point_limit(left_point, pic_width, pic_height)
        point_limit(right_point, pic_width, pic_height)
        point_limit(heigth_point, pic_width, pic_height)
        point_limit(low_point, pic_width, pic_height)
        left_point[1] += 1
        heigth_point[1] += 1
        print(left_point, right_point, heigth_point, low_point)

        # # 左边点变为左上角，右边点变为右下角，,像素越往右，越往下越大
        # if left_point[1] <= right_point[1]:  # 正角度
        #     new_right_point = [right_point[0], low_point[1]]
        #     pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        #     pts1 = np.float32([left_point, heigth_point, right_point])
        #     # 仿射变换，坐标变换
        #     M = cv2.getAffineTransform(pts1, pts2)
        #     dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
        #     point_limit(new_right_point)
        #     point_limit(heigth_point)
        #     point_limit(left_point)
        #     card_img = dst[
        #         int(new_right_point[1]): int(left_point[1]),
        #         int(left_point[0]): int(new_right_point[0]),
        #     ]
        #     card_imgs.append(card_img)
        # if left_point[1] > right_point[1]:  # 负角度
        #     new_left_point=[left_point[0], low_point[1]]
        #     pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        #     pts1 = np.float32([left_point, heigth_point, right_point])
        #     # 仿射变换，坐标变换
        #     M = cv2.getAffineTransform(pts1, pts2)
        #     dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
        #     point_limit(right_point)
        #     point_limit(heigth_point)
        #     point_limit(new_left_point)
        #     card_img = dst[
        #         int(new_left_point[1]): int(right_point[1]),
        #         int(new_left_point[0]): int(right_point[0]),
        #     ]
        #     card_imgs.append(card_img)

        if (left_point[1] <= right_point[1]) & (low_point[0] <= heigth_point[0]):  # 正角度
            new_right_point = [right_point[0], low_point[1]]
            new_low_point = [left_point[0], low_point[1]]
            # new_heigth_point=[left_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, new_low_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, low_point, right_point])
            # 仿射变换，坐标变换
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
            point_limit(new_right_point, pic_width, pic_height)
            point_limit(new_low_point, pic_width, pic_height)
            point_limit(left_point, pic_width, pic_height)
            card_img = dst[
                int(low_point[1]): int(left_point[1]),
                int(left_point[0]): int(right_point[0]),
            ]
            card_imgs.append(card_img)

        elif (left_point[1] > right_point[1]) & (low_point[0] > heigth_point[0]):  # 负角度

            new_left_point = [left_point[0], low_point[1]]
            new_low_point = [right_point[0], low_point[1]]
            # new_heigth_point=[right_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, new_low_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, low_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
            point_limit(right_point, pic_width, pic_height)
            point_limit(new_low_point, pic_width, pic_height)
            point_limit(new_left_point, pic_width, pic_height)
            card_img = dst[
                int(low_point[1]): int(right_point[1]),
                int(left_point[0]): int(right_point[0]),
            ]
            card_imgs.append(card_img)

        elif (left_point[1] > right_point[1]) & (low_point[0] < heigth_point[0]):  # 负角度
            new_right_point = [right_point[0], low_point[1]]
            new_low_point = [left_point[0], low_point[1]]
            # new_heigth_point=[right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, new_low_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, low_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
            point_limit(left_point, pic_width, pic_height)
            point_limit(new_low_point, pic_width, pic_height)
            point_limit(new_right_point, pic_width, pic_height)
            card_img = dst[
                int(low_point[1]): int(left_point[1]),
                int(left_point[0]): int(right_point[0]),
            ]
            card_imgs.append(card_img)

        elif (left_point[1] < right_point[1]) & (low_point[0] > heigth_point[0]):  # 负角度
            new_left_point = [left_point[0], low_point[1]]
            new_low_point = [right_point[0], low_point[1]]
            # new_heigth_point=[right_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, new_low_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, low_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
            point_limit(right_point, pic_width, pic_height)
            point_limit(new_low_point, pic_width, pic_height)
            point_limit(new_left_point, pic_width, pic_height)
            card_img = dst[
                int(low_point[1]): int(right_point[1]),
                int(left_point[0]): int(right_point[0]),
            ]
            card_imgs.append(card_img)
    # 显示所有框选的区域
    i = 0
    # cv2.imshow("0", card_imgs[0])
    for card_img in card_imgs:
        cv2.imshow(str(i), card_img)
        i = i + 1
    print("矩形区域定位后，疑似车牌数量：", len(card_imgs))

    # 7.增加颜色定位。针对疑似车牌区域，开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        # 有转换失败的可能，原因来自于上面矫正矩形出错。转换失败就跳过这个card
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                    yello += 1
                elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                    green += 1
                elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                    blue += 1
                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"

        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34  # 有的图片有色偏偏绿
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124  # 有的图片有色偏偏紫
        elif black + white >= card_img_count * 0.7:  # TODO
            color = "bw"
        colors.append(color)
        print(color)
        # print(blue, green, yello, black, white, card_img_count)
        # cv2.imshow(str(card_index), card_img)
        # cv2.waitKey(0)

        if limit1 == 0:
            continue
        # 以上为确定车牌颜色
        # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        card_imgs[card_index] = (
            card_img[yl:yh, xl:xr]
            if color != "green" or yl < (yh - yl) // 4
            else card_img[yl - (yh - yl) // 4: yh, xl:xr]
        )
        if need_accurate:  # 可能x或y方向未缩小，需要再试一次
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = (
            card_img[yl:yh, xl:xr]
            if color != "green" or yl < (yh - yl) // 4
            else card_img[yl - (yh - yl) // 4: yh, xl:xr]
        )
    car_plates = []
    plate_colors = []
    for i, color in enumerate(colors):
        if color in ("blue", "yello", "green"):
            car_plates.append(card_imgs[i])
            plate_colors.append(color)
            cv2.imshow(str(i) + str(color), card_imgs[i])
    print("矩形区域定位+颜色筛查后，疑似车牌数量：", len(plate_colors))
    return car_plates, plate_colors


######### 去除车牌无用部分##########


def find_waves(threshold, histogram):
    """根据设定的阈值和图片直方图，找出波峰，用于分隔字符"""
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_upanddown_border(plate_binary_img):
    """去除车牌上下无用的边缘部分，确定上下边界"""
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    # 挑选跨度最大的波峰
    wave_span = 0.0
    selected_wave = []
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]: selected_wave[1], :]
    return plate_binary_img


# 分割车牌
##################### 二分-K均值聚类算法############################


def distEclud(vecA, vecB):
    """
    计算两个坐标向量之间的街区距离
    """
    return np.sum(abs(vecA - vecB))


def randCent(dataSet, k):
    n = dataSet.shape[1]  # 列数
    centroids = np.zeros((k, n))  # 用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:, j], axis=0)
        rangeJ = float(np.max(dataSet[:, j])) - minJ
        for i in range(k):
            centroids[i:, j] = minJ + rangeJ * (i + 1) / k
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros(
        (m, 2)
    )  # 这个簇分配结果矩阵包含两列，一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的街区距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    这个函数首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    输入：dataSet是一个ndarray形式的输入数据集
          k是用户指定的聚类后的簇的数目
         distMeas是距离计算函数
    输出:  centList是一个包含类质心的列表，其中有k个元素，每个元素是一个元组形式的质心坐标
            clusterAssment是一个数组，第一列对应输入数据集中的每一行样本属于哪个簇，第二列是该样本点与所属簇质心的距离
    """
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroid0 = np.mean(dataSet, axis=0).tolist()
    centList = []
    centList.append(centroid0)
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.array(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:  # 小于K个簇时
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0] == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(
                clusterAssment[np.nonzero(clusterAssment[:, 0] != i), 1]
            )
            if (sseSplit + sseNotSplit) < lowestSSE:  # 如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0, :].tolist()
        centList.append(bestNewCents[1, :].tolist())
        clusterAssment[
            np.nonzero(clusterAssment[:, 0] == bestCentTosplit)[0], :
        ] = bestClustAss
    return centList, clusterAssment


def split_licensePlate_character(plate_binary_img):
    """
    此函数用来对车牌的二值图进行水平方向的切分，将字符分割出来
    输入： plate_gray_Arr是车牌的二值图，rows * cols的数组形式
    输出： character_list是由分割后的车牌单个字符图像二值图矩阵组成的列表
    """
    plate_binary_Arr = np.array(plate_binary_img)
    row_list, col_list = np.nonzero(plate_binary_Arr >= 255)
    dataArr = np.column_stack((col_list, row_list))  # dataArr的第一列是列索引，第二列是行索引，要注意
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])
    split_list = []
    for centroids_ in centroids_sorted:
        i = centroids.index(centroids_)
        current_class = dataArr[np.nonzero(clusterAssment[:, 0] == i)[0], :]
        x_min, y_min = np.min(current_class, axis=0)
        x_max, y_max = np.max(current_class, axis=0)
        split_list.append([y_min, y_max, x_min, x_max])
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]: split_list[i][3]]
        character_list.append(single_character_Arr)
        cv2.imshow("character" + str(i), single_character_Arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return character_list  # character_list中保存着每个字符的二值图数据

    ############################
    # 测试用
    # print(col_histogram )
    # fig = plt.figure()
    # plt.hist( col_histogram )
    # plt.show()
    ############################


# if __name__ == "__main__":
#     cfg = json.load(open("config.js"))
#     CFG = cfg[0]
#     trainSvm = recognition.TRAINSVM()
#     trainSvm.train_svm()
#     train_path = r"carplate\train"
#     correct = 0
#     count = 0
#     for filename in os.listdir(train_path):
#         if filename.endswith('jpg') or filename.endswith('png'):
#             print(filename)
#             img_path = os.path.join(train_path, filename)
#             img_label = os.path.splitext(filename)  # 后缀
#             img_label = list(img_label[0])
#             car_plates, plate_colors = pretreatment(img_path)
#             for i, plate_color in enumerate(plate_colors):
#                 # 做一次锐化处理
#                 kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
#                 card_img = cv2.filter2D(car_plates[i], -1, kernel=kernel)

#                 # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
#                 # if plate_color == "green" or plate_color == "yello":
#                 #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
#                 # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # character_list = split_licensePlate_character(plate_gray_img)

#                 # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
#                 # if plate_color == "green" or plate_color == "yello":
#                 #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
#                 # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # plate_gray_img = remove_upanddown_border(plate_gray_img)
#                 # cv2.imshow("remove_upanddown", plate_gray_img)
#                 # character_list = split_licensePlate_character(plate_gray_img)

#                 # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
#                 # if plate_color == "green" or plate_color == "yello":
#                 #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
#                 # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # temp.split_char2(plate_gray_img)

#                 # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
#                 # if plate_color == "green" or plate_color == "yello":
#                 #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
#                 # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # plate_gray_img = remove_upanddown_border(plate_gray_img)
#                 # cv2.imshow("remove_upanddown", plate_gray_img)
#                 # temp.split_char2(plate_gray_img)

#                 plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
#                 if plate_color == "green" or plate_color == "yello":
#                     plate_gray_img = cv2.bitwise_not(plate_gray_img)
#                 ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 part_cards = temp.split_char2(plate_gray_img)
#                 if len(part_cards)==0:
#                     print("分割失败")
#                     count += 1
#                     continue
#                 # print("==============================")
#                 # print(part_cards)
#                 predict_result = trainSvm.recog_card(part_cards)
#                 print(predict_result, " ", plate_color)
#                 if img_label == predict_result:
#                     correct += 1
#         count += 1
#     print("train_img正确率：", correct / count)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = json.load(open("config.js"))
    CFG = cfg[0]
    trainSvm = recognition.TRAINSVM()
    trainSvm.train_svm()
    train_path = r"carplate\train"
    correct = 0
    count = 0
    img_path = r"D:\1MyLearningData\LPR\hwtg\carplate\train\沪JS6999.jpg"

    # resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
    # for resize_rate in resize_rates:
    #     print("resize_rate:", resize_rate)
    #     car_plates, plate_colors = pretreatment(img_path, resize_rate)
    #     print(plate_colors)
    #     if len(plate_colors):
    #         break

    car_plates, plate_colors = pretreatment(img_path)
    for i, plate_color in enumerate(plate_colors):
        # 做一次锐化处理
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        card_img = cv2.filter2D(car_plates[i], -1, kernel=kernel)

        # # Kmeans去除车牌铆钉
        # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
        # if plate_color == "green" or plate_color == "yello":
        #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
        # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # #去除车牌铆钉
        # # plate_gray_img = remove_upanddown_border(plate_gray_img)
        # # cv2.imshow("remove_upanddown", plate_gray_img)
        # character_list = split_licensePlate_character(plate_gray_img)

        # # 去除车牌铆钉，投影分割
        # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
        # if plate_color == "green" or plate_color == "yello":
        #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
        # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # #去除车牌铆钉
        # # plate_gray_img = remove_upanddown_border(plate_gray_img)
        # # cv2.imshow("remove_upanddown", plate_gray_img)
        # temp.split_char2(plate_gray_img)

        # 去除车牌铆钉，波峰分割
        plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
        if plate_color == "green" or plate_color == "yello":
            plate_gray_img = cv2.bitwise_not(plate_gray_img)
        ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plate_gray_img = remove_upanddown_border(plate_gray_img)
        # cv2.imshow("remove_upanddown", plate_gray_img)
        part_cards = temp.split_char3(plate_gray_img)

        # 字符识别
        # plate_gray_img = cv2.cvtColor(car_plates[i], cv2.COLOR_BGR2GRAY)
        # if plate_color == "green" or plate_color == "yello":
        #     plate_gray_img = cv2.bitwise_not(plate_gray_img)
        # ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # part_cards = temp.split_char2(plate_gray_img)
        # # print("==============================")
        # # print(part_cards)

        # predict_result = trainSvm.recog_card(part_cards)
        # print(predict_result, " ", plate_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
