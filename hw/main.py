import cv2
import json
import numpy as np
from PIL import Image, ImageTk

MAX_WIDTH = 1000  # 原始图片最大宽度
MIN_AREA = 2000  # 车牌区域允许最大面积
j = json.load(open("config.js"))
for c in j["config"]:
    if c["open"]:
        CFG = c.copy()
        break


def preprocess(car_pic, resize_rate=1):
    img = car_pic
    #########################################
    cv2.imshow("Src", img)
    pic_hight, pic_width = img.shape[:2]
    # 按照宽度大小进行等比缩放
    if pic_width > MAX_WIDTH:
        pic_rate = MAX_WIDTH / pic_width
        img = cv2.resize(
            img,
            (MAX_WIDTH, int(pic_hight * pic_rate)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        # modified
        pic_hight, pic_width = img.shape[:2]
    # 按照设置的比例进行缩放
    if resize_rate != 1:
        img = cv2.resize(
            img,
            (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        pic_hight, pic_width = img.shape[:2]

    print("resized h:", pic_hight, "w:", pic_width)
    blur = CFG["blur"]
    # 高斯去噪
    if blur > 0:
        # 图片分辨率调整
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    #################################################
    cv2.imshow("Gauss", img)
    # 保存原图
    oldimg = img
    # 转换灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #################################################
    cv2.imshow("Gray", img)
    # 去掉图像中不会是车牌的区域
    kernel = np.ones((20, 20), np.uint8)
    # 利用形态学变换 腐蚀 膨胀等操作去除噪声
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #################################################
    cv2.imshow("MorphologyEx", img_opening)
    # 按照比例混合灰度图像与形态变换结果 突出车牌区域
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
    #################################################
    cv2.imshow("Mixed", img_opening)
    # 找到图像边缘
    # 二值化
    ret, img_thresh = cv2.threshold(
        img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    #################################################
    cv2.imshow("thresh", img_thresh)
    # 边缘
    img_edge = cv2.Canny(img_thresh, 100, 200)
    #################################################
    cv2.imshow("Canny", img_edge)
    # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((CFG["morphologyr"], CFG["morphologyc"]), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    #################################################
    cv2.imshow("Morph close", img_edge1)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    #################################################
    cv2.imshow("Morph open", img_edge2)
    # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    try:
        contours, hierarchy = cv2.findContours(
            img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    except ValueError:
        image, contours, hierarchy = cv2.findContours(
            img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    print("Car labels area:", len(contours))

    # 一一排除不是车牌的矩形区域
    car_contours = []
    for cnt in contours:
        # 寻找点集的最小矩形区域
        rect = cv2.minAreaRect(cnt)
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            cv2.imshow("Aimed area", oldimg)
    print("Confirmed car labels area:", len(car_contours))
    return car_contours, pic_hight, pic_width


def localization(car_contours, pic_hight, pic_width):
    print("Begin to localization...")
    card_imgs = []
    # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    for rect in car_contours:
        # 创造角度，使得左、高、右、低拿到正确的值
        if rect[2] > -1 and rect[2] < 1:
            angle = 1
        else:
            angle = rect[2]
        # 扩大范围，避免车牌边缘被排除
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)

        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        # 避免扩大范围后的车牌超出图片 规范车牌所在范围为图形内部
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        #  # 正角度
        # if left_point[1] <= right_point[1]:
        #     new_right_point = [right_point[0], heigth_point[1]]
        #     pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        #     pts1 = np.float32([left_point, heigth_point, right_point])
        #     M = cv2.getAffineTransform(pts1, pts2)
        #     dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
        #     point_limit(new_right_point)
        #     point_limit(heigth_point)
        #     point_limit(left_point)
        #     card_img = dst[
        #         int(left_point[1]) : int(heigth_point[1]),
        #         int(left_point[0]) : int(new_right_point[0]),
        #     ]
        #     card_imgs.append(card_img)
        #     # cv2.imshow("card", card_img)
        #     # cv2.waitKey(0)
        # elif left_point[1] > right_point[1]:  # 负角度

        #     new_left_point = [left_point[0], heigth_point[1]]
        #     pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        #     pts1 = np.float32([left_point, heigth_point, right_point])
        #     M = cv2.getAffineTransform(pts1, pts2)
        #     dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
        #     point_limit(right_point)
        #     point_limit(heigth_point)
        #     point_limit(new_left_point)
        #     card_img = dst[
        #         int(right_point[1]) : int(heigth_point[1]),
        #         int(new_left_point[0]) : int(right_point[0]),
        #     ]
        #     card_imgs.append(card_img)
        #     # cv2.imshow("card", card_img)
        #     # cv2.waitKey(0)


def predict(car_pic, resize_rate=1):
    car_contours, pic_hight, pic_width = preprocess(car_pic, resize_rate)
    localization(car_contours, pic_hight, pic_width)
    cv2.waitKey(0)


def imreadex(filename):
    img_bgr = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
    for resize_rate in resize_rates:
        print("resize_rate:", resize_rate)
        r, roi, color = predict(img_bgr, resize_rate)
        if r:
            print(r)
            break


filename = r"D:\1MyLearningData\LPR\License-Plate-Recognition\test\car3.jpg"
imreadex(filename)
