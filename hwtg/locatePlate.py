import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import temp
import recognition
import os

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1200  # 原始图片最大宽度
MIN_AREA = 800  # 车牌区域允许最大面积
PROVINCE_START = 1000

""" 
限制裁剪车牌的点坐标，不能为负值
"""


def point_limit(point, pic_width, pic_height):
    if point[0] < 0:
        point[0] = 0
    elif point[0] > pic_width:
        point[0] = pic_width
    if point[1] < 0:
        point[1] = 0
    elif point[1] > pic_height:
        point[1] = pic_height


""" 
通过颜色，边缘化定位到车牌后，精确定位裁剪车牌
"""


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    # col_num_limit = self.cfg["col_num_limit"]
    row_num_limit = row_num * 0.6
    col_num_limit = col_num * 0.6 if color != "green" else col_num * 0.5  # 绿色有渐变
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
        if count > row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


""" 
根据找出的波峰，分隔图片，从而得到逐个字符图片
"""


def seperate_card(img, waves):

    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]: wave[1]])
    return part_cards


def stretching(img_gray):
    """ 灰度拉伸 """
    maxi = float(img_gray.max())
    mini = float(img_gray.min())
    if maxi != mini:
        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                img_gray[i, j] = 255 / (maxi - mini) * img_gray[i, j] - (255 * mini) / (maxi - mini)
    img_stretched = img_gray
    return img_stretched


""" 
颜色定位，开闭操作
"""


def color_oc_operation(hsv_gray):
    # 1.灰度图的高斯平滑去噪
    gray_gauss = cv2.GaussianBlur(hsv_gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    gray_gauss = stretching(gray_gauss)
    
    # 2.开运算，与上次运算结果融合
    kernel = np.ones((5, 5), np.uint8)
    img_opening = cv2.morphologyEx(gray_gauss, cv2.MORPH_OPEN, kernel)

    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("img_opening0",gray_gauss)
    # cv2.imshow("img_opening1",img_opening)

    # kernel = np.ones((3, 3), np.uint8)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_OPEN, kernel)

    # kernel = np.ones((5, 5), np.uint8)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    # img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("img_opening2",img_opening)

    # # cv2.imshow("img_opening0",gray_gauss)
    # # cv2.imshow("img_opening1",img_opening)

    # # cv2.imshow("img_opening2",img_opening)
    # 3.二值化，并Canny边缘化(或不边缘)
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = img_thresh
    # cv2.imshow("img_thresh",img_thresh)

    # 4.使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_OPEN, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    
    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


""" 
边缘检测定位，开闭操作
"""


def edge_oc_operation(hsv_gray):
    # 找车牌轮廓
    # 1.高斯平滑去噪
    img_gauss = cv2.GaussianBlur(hsv_gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 1.2灰度拉伸
    img_gauss = stretching(img_gauss)
    
    
    
    # # 2.开运算，与上次运算结果融合
    # kernel = np.ones((25, 25), np.uint8)
    # img_opening = cv2.morphologyEx(img_gauss, cv2.MORPH_OPEN, kernel)
    # # cv2.imshow("img_gauss",img_gauss)
    # # cv2.imshow("img_opening0",img_opening)
    # img_opening = cv2.addWeighted(img_gauss, 1, img_opening, -1, 0)
    # # cv2.imshow("img_opening1",img_opening)
    # # 3.二值化，找到图像边缘
    # ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_edge = cv2.Canny(img_thresh, img_thresh.shape[0], img_thresh.shape[1])
    # # img_edge = cv2.Canny(img_thresh, 100, 200)
    # # cv2.imshow("img_thresh",img_thresh)

    # # 4.使用开运算和闭运算让图像边缘成为一个整体
    # # kernel = np.ones((15,60), np.uint8)
    # # img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    # # img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_OPEN, kernel)
    # # cv2.imshow("img_edge",img_edge)

    # kernel = np.ones((5, 25), np.uint8)
    # img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    # img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    
    # kernel = np.ones((11, 6), np.uint8)
    # img_edge2 = cv2.morphologyEx(img_edge2, cv2.MORPH_OPEN, kernel)
    
    
    
    # 2、顶帽运算
    # gray = cv2.equalizeHist(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    tophat = cv2.morphologyEx(img_gauss, cv2.MORPH_TOPHAT, kernel)
    # cv2.imshow('tophat', tophat)
 
    # 3、Sobel算子提取y方向边缘（揉成一坨）
    y = cv2.Sobel(tophat, cv2.CV_16S, 1,     0)
    absY = cv2.convertScaleAbs(y)
    # cv2.imshow('absY', absY)
 
    # 4、自适应二值化（阈值自己可调）
    ret, binary = cv2.threshold(absY, 75, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary', binary)
 
    # 5、开运算分割（纵向去噪，分隔）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    Open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('Open', Open)
 
    # 6、闭运算合并，把图像闭合、揉团，使图像区域化，便于找到车牌区域，进而得到轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 15))
    close = cv2.morphologyEx(Open, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('close', close)
 
    # 7、膨胀/腐蚀（去噪得到车牌区域）
    # 中远距离车牌识别
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    # 近距离车牌识别
    # kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (79, 15))
    # kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    # 7-1、腐蚀、膨胀（去噪）
    erode_y = cv2.morphologyEx(close, cv2.MORPH_ERODE, kernel_y)
    # cv2.imshow('erode_y', erode_y)
    dilate_y = cv2.morphologyEx(erode_y, cv2.MORPH_DILATE, kernel_y)
    # cv2.imshow('dilate_y', dilate_y)
    # 7-1、膨胀、腐蚀（连接）（二次缝合）
    dilate_x = cv2.morphologyEx(dilate_y, cv2.MORPH_DILATE, kernel_x)
    # cv2.imshow('dilate_x', dilate_x)
    erode_x = cv2.morphologyEx(dilate_x, cv2.MORPH_ERODE, kernel_x)
    # cv2.imshow('erode_x', erode_x)
 
    # 8、腐蚀、膨胀：去噪
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    erode = cv2.morphologyEx(erode_x, cv2.MORPH_ERODE, kernel_e)
    # cv2.imshow('erode', erode)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
    dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, kernel_d)
    # cv2.imshow('dilate', dilate)
    img_edge2=dilate
    # cv2.imshow("img_edge1",img_edge1)
    # cv2.imshow("img_edge2",img_edge2)
    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


""" 
矫正图片，摆正
"""


def rectify_rec(car_contours, oldimg, pic_width, pic_height):
    # 6.矫正矩形区域。矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    card_imgs = []
    for i, rect in enumerate(car_contours):
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 10, rect[1][1] + 10), angle)  # 扩大范围，避免车牌边缘被排除
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

        # # 左边点变为左上角，右边点变为右下角，,像素越往右，越往下越大
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
    return card_imgs


""" 
利用颜色重新裁剪图片，重新定位
"""


def color_relocate(card_imgs):
    # 7.增加颜色定位。针对疑似车牌区域，开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yellow = blue = black = white = 0
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
                    yellow += 1
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
        if yellow * 2.5 >= card_img_count:
            color = "yellow"
            limit1 = 11
            limit2 = 34  # 有的图片有色偏偏绿
        elif green * 2.5 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2.5 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124  # 有的图片有色偏偏紫
        elif black + white >= card_img_count * 0.7:  # TODO
            color = "bw"
        colors.append(color)
        print(color)
        # print(blue, green, yellow, black, white, card_img_count)
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
    return colors


"""
定位车牌，并提取车车牌。
思路：颜色定位，颜色再定位
输入：filename图片地址，resize_rate放缩图片的比例
输出：定位后，截取后的车牌。车牌颜色。可能输出不止一个车牌，需要在识别中判断。
"""


def color_locate1(res_id, res_hsv, res_name, oldimg, pic_width, pic_height):
    # 将提取对应颜色的图片转为灰度图
    hsv_gray = cv2.cvtColor(res_hsv, cv2.COLOR_BGR2GRAY)
    contours = color_oc_operation(hsv_gray)

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
            # 在原图上画框线
            oldimg1 = cv2.drawContours(oldimg1, [box], 0, (0, 0, 255), 2)
            cv2.imshow("oldimg1", oldimg1)

    # 6.矫正矩形区域，摆正
    card_imgs = rectify_rec(car_contours, oldimg, pic_width, pic_height)

    for i in range(len(card_imgs)):
        # 放大车牌区域图片
        card_imgs[i] = cv2.resize(card_imgs[i], (card_imgs[i].shape[:2][1] * 3, card_imgs[i].shape[:2][0] * 3), interpolation=cv2.INTER_AREA)

    print(res_name[res_id] + " rec plate:", len(card_imgs))
    # 7.增加颜色定位。针对疑似车牌区域，开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    colors = color_relocate(card_imgs)

    car_plates = []
    plate_colors = []
    for i, color in enumerate(colors):
        if (color in ("blue", "yellow", "green")) & (color == res_name[res_id]):
            car_plates.append(card_imgs[i])
            plate_colors.append(color)
    print(res_name[res_id] + " rec recolor plate:", len(plate_colors))

    return car_plates, plate_colors


"""
定位车牌，并提取车车牌。
思路：边缘检测，颜色再定位
输入：filename图片地址，resize_rate放缩图片的比例
输出：定位后，截取后的车牌。车牌颜色。可能输出不止一个车牌，需要在识别中判断。
"""


def edge_locate1(res_hsv, oldimg, pic_width, pic_height):
    # 将提取对应颜色的图片转为灰度图
    hsv_gray = cv2.cvtColor(res_hsv, cv2.COLOR_BGR2GRAY)

    contours = edge_oc_operation(hsv_gray)

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
            # 在原图上画框线
            oldimg1 = cv2.drawContours(oldimg1, [box], 0, (0, 0, 255), 2)
            cv2.imshow("oldimg1", oldimg1)

    # 6.矫正矩形区域，摆正
    card_imgs = rectify_rec(car_contours, oldimg, pic_width, pic_height)

    for i in range(len(card_imgs)):
        # 放大车牌区域图片
        card_imgs[i] = cv2.resize(card_imgs[i], (card_imgs[i].shape[:2][1] * 3, card_imgs[i].shape[:2][0] * 3), interpolation=cv2.INTER_AREA)

    print("rec plate:", len(card_imgs))
    # 7.增加颜色定位。针对疑似车牌区域，开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    # colors = color_relocate(card_imgs)
    colors = []
    for i in range(len(card_imgs)):
        colors.append("blue")

    car_plates = []
    plate_colors = []
    for i, color in enumerate(colors):
        if color in ("blue", "yellow", "green"):
            car_plates.append(card_imgs[i])
            plate_colors.append(color)
    print("rec recolor plate:", len(plate_colors))

    return car_plates, plate_colors
