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


def mark_zone_color(img):
    # 根据颜色在原始图像上标记
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 图像二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #drawing = img
    # cv2.drawContours(drawing, contours, -1, (0, 0, 255), 3)  # 填充轮廓颜色
    #cv2.imshow('drawing', drawing)
    # cv2.waitKey(0)
    # print(contours)

    temp_contours = []  # 存储合理的轮廓
    car_plates = []
    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) > MIN_AREA:
                temp_contours.append(contour)
            car_plates = []
            for temp_contour in temp_contours:
                rect_tupple = cv2.minAreaRect(temp_contour)
                rect_width, rect_height = rect_tupple[1]
                if rect_width < rect_height:
                    rect_width, rect_height = rect_height, rect_width
                aspect_ratio = rect_width / rect_height
                # 车牌正常情况下宽高比在2 - 5.5之间
                if aspect_ratio > 2 and aspect_ratio < 5.5:
                    car_plates.append(temp_contour)
                    rect_vertices = cv2.boxPoints(rect_tupple)
                    rect_vertices = np.int0(rect_vertices)
            if len(car_plates) == 1:
                oldimg = cv2.drawContours(img, [rect_vertices], -1, (0, 0, 255), 2)
                #cv2.imshow("che pai ding wei", oldimg)
                # print(rect_tupple)
                break

    # 把车牌号截取出来
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(img, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
            card_img = img[col_min:col_max, row_min:row_max, :]
            cv2.imshow("img", img)
        cv2.imshow("card_img.", card_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 1
    return 0

# 定位车牌


def color_position(img):
    colors = [([26, 43, 46], [34, 255, 255]),  # 黄色
              ([100, 43, 46], [124, 255, 255]),  # 蓝色
              ([35, 43, 46], [77, 255, 255])  # 绿色
              ]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for (lower, upper) in colors:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应的颜色
        mask = cv2.inRange(hsv, lowerb=lower, upperb=upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("output", output)
        k = mark_zone_color(output)
        if k == 1:
            return 1
        # 展示图片
        #cv2.imshow("image", img)
        #cv2.imshow("image-color", output)
        # cv2.waitKey(0)
    return 0


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

if __name__ == "__main__":
    resize_rate=1
    imgPath = r"D:\1MyLearningData\LPR\hwtg\carplate\train\川A82M83.jpg"
    img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 调整图片大小
    # 获取输入图像的高和宽
    pic_height, pic_width = img.shape[:2]
    if pic_width > MAX_WIDTH:
        change_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_height * change_rate)), interpolation=cv2.INTER_AREA,)
        # 获取输入图像的高和宽
        pic_height, pic_width = img.shape[:2]

    if resize_rate != 1:
        img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_height * resize_rate)), interpolation=cv2.INTER_LANCZOS4)
        pic_height, pic_width = img.shape[:2]


    oldimg=img.copy()
    colors = [([15, 55, 55], [50, 255, 255]),  # 黄色
              ([100, 43, 46], [124, 255, 255]),  # 蓝色
              ([0, 3, 116], [76, 211, 255])]  # 绿色

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, np.array(colors[0][0]), np.array(colors[0][1]))
    mask_blue = cv2.inRange(hsv, np.array(colors[1][0]), np.array(colors[1][1]))
    mask_green = cv2.inRange(hsv, np.array(colors[2][0]), np.array(colors[2][1]))

    res_yellow = cv2.bitwise_and(hsv, hsv, mask=mask_yellow)
    res_blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    res_green = cv2.bitwise_and(hsv, hsv, mask=mask_green)
    cv2.imshow("output1", res_blue)
    # 根据阈值找到对应颜色
    output = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    cv2.imshow("output2", output)

    #二值化
    ret,thresh1 = cv2.threshold(output,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('gray',gray)
    #闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3))  
    img_opening = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel) 
    cv2.imshow("img_opening", img_opening) 

    # # 1.灰度图的高斯平滑去噪
    # gray_img_gauss = cv2.GaussianBlur(output, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # cv2.imshow("output3_gauss", output)

    # # 2.开运算，与上次运算结果融合
    # kernel = np.ones((20, 20), np.uint8)
    
    # gray_img_gauss = cv2.morphologyEx(gray_img_gauss, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("img_opening", gray_img_gauss)
    # img_opening = cv2.morphologyEx(gray_img_gauss, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("img_opening0", img_opening)
    # # img_opening = cv2.addWeighted(gray_img_gauss, 1, img_opening, -1, 0)
    # # cv2.imshow("img_opening", img_opening)
    
    # # # 3.二值化，找到图像边缘
    # # ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # img_edge = cv2.Canny(img_thresh, 100, 200)
    # # cv2.imshow("img_edge", img_edge)

    # # kernel = np.ones((5, 20), np.uint8)
    # # img_edge1 = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    # # cv2.imshow("img_edge1", img_edge1)
    # # img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # # cv2.imshow("img_edge2", img_edge2)

    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
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
    for rect in car_contours:
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] , rect[1][1] ), angle)  # 扩大范围，避免车牌边缘被排除
        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_height]
        # 定位车牌最左，最右，最高，最低的点（车牌可能是不规则矩形）,像素越往右，越往下越大
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point
        # 左边点变为左上角，右边点变为右下角，,像素越往右，越往下越大
        new_left_point=[left_point[0], low_point[1]]
        new_right_point = [right_point[0], heigth_point[1]]
        pts2 = np.float32([new_left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        pts1 = np.float32([left_point, heigth_point, right_point])
        # 仿射变换，坐标变换
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
        point_limit(new_right_point)
        point_limit(heigth_point)
        point_limit(new_left_point)
        card_img = dst[
            int(new_left_point[1]): int(new_right_point[1]),
            int(new_left_point[0]): int(new_right_point[0]),
        ]
        card_imgs.append(card_img)
    # 显示所有框选的区域
    i = 0
    for card_img in card_imgs:
        cv2.imshow(str(i), card_img)
        i = i + 1
    print("矩形区域定位后，疑似车牌数量：", len(card_imgs))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
