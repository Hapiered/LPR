import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
MIN_AREA = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


""" 预处理。获取已经矫正，或分区域后的车牌
    输入：图片彩色图或者图片地址
    输出：处理后的车牌，车牌颜色
"""


def pretreatment(filename):
    imgSrc = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
    # 调整图片大小
    rows, cols = imgSrc.shape[:2]  # 获取输入图像的高和宽
    pic_height, pic_width = rows, cols
    img_copy = imgSrc.copy()
    if pic_width > MAX_WIDTH:
        change_rate = MAX_WIDTH / pic_width
        imgResize = cv2.resize(imgSrc, (MAX_WIDTH, int(pic_height * change_rate)), interpolation=cv2.INTER_AREA)
        img_copy = imgResize.copy()
    # 找车牌轮廓
    # 1.转灰度图，高斯平滑去噪
    if CFG["blur"] > 0:
        img_gauss = cv2.GaussianBlur(img_copy, (CFG["blur"], CFG["blur"]), 0, 0, cv2.BORDER_DEFAULT)
    oldimg = img_gauss
    gray_img_gauss = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)
    # 2.开运算，与上次运算结果融合
    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(gray_img_gauss, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img_gauss, 1, img_opening, -1, 0)
    # 3.二值化，找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # 4.使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((CFG["morphologyr"], CFG["morphologyc"]), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 6.逐个排除不是车牌的矩形区域，获取外接矩阵。（1）车牌最小面积，（2）车牌宽高比范围
    # 6.1（1）车牌最小面积
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    # 6.2（2）车牌宽高比范围
    car_contours = []
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
            oldimg1 = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            cv2.imshow("oldimg1", oldimg1)
    # 6.矫正矩形区域。矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    card_imgs = []
    for rect in car_contours:
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_height]
        # 定位车牌最左，最右，最高，最低的点（车牌可能是不规则矩形）
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point
        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            # 仿射变换，坐标变换
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)
        elif left_point[1] > right_point[1]:  # 负角度
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)
    # # 显示所有框选的区域
    # i = 0
    # for card_img in card_imgs:
    #     cv2.imshow(str(i), card_img)
    #     i = i + 1
    print("矩形区域定位后，疑似车牌数量：", len(card_imgs))

    # 7.增加颜色定位。针对疑似车牌区域，开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    # 利用颜色排除不是车牌的区域，并进行较精细定位
    colors = []
    lisencePlates = []
    for img in card_imgs:
        green = yellow = blue = 0
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        row, col = hsvImg.shape[:2]
        pixelCount = row * col
        # print("pixel count: ", pixelCount)

        for i in range(row):
            for j in range(col):
                H = hsvImg.item(i, j, 0)
                S = hsvImg.item(i, j, 1)
                V = hsvImg.item(i, j, 2)
                if 15 <= H <= 30 and S >= 60 and V >= 45:
                    yellow += 1
                elif 65 <= H <= 90 and S >= 15 and V >= 65:
                    green += 1
                elif 100 <= H <= 120 and S >= 55 and V >= 40:
                    blue += 1
        if yellow == max(yellow, blue, green) and yellow > pixelCount * 0.35:
            xl, xr, yh, yl = fineMap("yellow", hsvImg)
            img = img[yh:yl, xl:xr]
            h, w = img.shape[:2]
            if 2.2 < w / h < 5.3:
                colors.append("yellow")
                lisencePlates.append(img)
                # imshow(str(img), img)
        if blue == max(yellow, blue, green) and blue > pixelCount * 0.35:
            xl, xr, yh, yl = fineMap("blue", hsvImg)
            img = img[yh:yl, xl:xr]
            h, w = img.shape[:2]
            if 2.2 < w / h < 5.3:
                lisencePlates.append(img)
                colors.append("blue")
                cv2.imshow(str(img), img)
        if green == max(yellow, blue, green) and green > pixelCount * 0.35:
            xl, xr, yh, yl = fineMap("green", hsvImg)
            img = img[yl - int((yl - yh) * 4 / 3):yl, xl:xr]
            h, w = img.shape[:2]
            if 2.2 < w / h < 5.3:
                lisencePlates.append(img)
                colors.append("green")
                # imshow(str(img), img)
    print("num of lisence plate: " + str(len(lisencePlates)))
    
    return colors, lisencePlates

# 精细定位


def fineMap(color, hsvImg):
    if color == "yellow":
        limit1 = 15
        limit2 = 30
        limitS = 60
        limitV = 45
    elif color == "green":
        limit1 = 65
        limit2 = 90
        limitS = 15
        limitV = 65
    elif color == "blue":
        limit1 = 100
        limit2 = 120
        limitS = 55
        limitV = 40
    row_num, col_num = hsvImg.shape[:2]
    xl = col_num - 1
    xr = 0
    yh = row_num - 1
    yl = 0
    # col_num_limit = self.cfg["col_num_limit"]
    row_num_limit = row_num * 0.5 if color != "green" else row_num * 0.3  # 绿色有渐变
    col_num_limit = col_num * 0.3
    # 按行
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = hsvImg.item(i, j, 0)
            S = hsvImg.item(i, j, 1)
            V = hsvImg.item(i, j, 2)
            if limit1 <= H <= limit2 and S >= limitS and V >= limitV:
                count += 1
        if count > col_num_limit:
            if yh > i:
                yh = i
            if yl < i:
                yl = i
    # 按列
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = hsvImg.item(i, j, 0)
            S = hsvImg.item(i, j, 1)
            V = hsvImg.item(i, j, 2)
            if limit1 <= H <= limit2 and S >= limitS and V >= limitV:
                count += 1
        if count > row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


if __name__ == "__main__":
    cfg = json.load(open("config.js"))
    CFG = cfg[0]
    pretreatment("AEK882.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
