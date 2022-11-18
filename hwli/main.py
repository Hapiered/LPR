import cv2
import json
from numpy.linalg import norm
import numpy as np
# from PIL import Image, ImageTk

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
MIN_AREA = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000
# 可识别的省份
provinces = [
    "zh_cuan", "川", "zh_e", "鄂", "zh_gan", "赣", "zh_gan1", "甘", "zh_gui", "贵",
    "zh_gui1", "桂", "zh_hei", "黑", "zh_hu", "沪", "zh_ji", "冀", "zh_jin", "津",
    "zh_jing", "京", "zh_jl", "吉", "zh_liao", "辽", "zh_lu", "鲁", "zh_meng", "蒙",
    "zh_min", "闽", "zh_ning", "宁", "zh_qing", "靑", "zh_qiong", "琼", "zh_shan",
    "陕", "zh_su", "苏", "zh_sx", "晋", "zh_wan", "皖", "zh_xiang", "湘", "zh_xin",
    "新", "zh_yu", "豫", "zh_yu1", "渝", "zh_yue", "粤", "zh_yun", "云", "zh_zang",
    "藏", "zh_zhe", "浙"
]

j = json.load(open("config.js"))
for c in j["config"]:
    if c["open"]:
        CFG = c.copy()
        break


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


# SVM CV2 模型
class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


# 修复点的位置
def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 根据颜色修复车牌范围
def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
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
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [
            np.bincount(b.ravel(), m.ravel(), bin_n)
            for b, m in zip(bin_cells, mag_cells)
        ]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def preprocess(img, resize_rate=1):
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
    # cv2.imshow("Gauss", img)
    # 保存原图
    oldimg = img
    # 转换灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #################################################
    # cv2.imshow("Gray", img)
    # 去掉图像中不会是车牌的区域
    kernel = np.ones((20, 20), np.uint8)
    # 利用形态学变换 腐蚀 膨胀等操作去除噪声
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #################################################
    # cv2.imshow("MorphologyEx", img_opening)
    # 按照比例混合灰度图像与形态变换结果 突出车牌区域
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
    #################################################
    # cv2.imshow("Mixed", img_opening)
    # 找到图像边缘
    # 二值化
    ret, img_thresh = cv2.threshold(img_opening, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #################################################
    # cv2.imshow("thresh", img_thresh)
    # 边缘
    img_edge = cv2.Canny(img_thresh, 100, 200)
    #################################################
    # cv2.imshow("Canny", img_edge)
    # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((CFG["morphologyr"], CFG["morphologyc"]), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    #################################################
    # cv2.imshow("Morph close", img_edge1)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    #################################################
    # cv2.imshow("Morph open", img_edge2)
    # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
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
            #################################################
            # cv2.imshow("Aimed area", oldimg)
    print("Confirmed car labels area:", len(car_contours))
    return car_contours, pic_hight, pic_width, oldimg


def localization(car_contours, pic_hight, pic_width, oldimg):
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

    # 正角度
    if left_point[1] <= right_point[1]:
        new_right_point = [right_point[0], heigth_point[1]]
        pts2 = np.float32([left_point, heigth_point,
                           new_right_point])  # 字符只是高度需要改变
        pts1 = np.float32([left_point, heigth_point, right_point])
        # getAffineTransform通过确认源图像中不在同一直线的三个点对应的目标图像的位置
        # 来获取对应仿射变换矩阵，从而用该仿射变换矩阵对图像进行统一的仿射变换
        M = cv2.getAffineTransform(pts1, pts2)
        # warpAffine 旋转仿射变换
        dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
        point_limit(new_right_point)
        point_limit(heigth_point)
        point_limit(left_point)
        card_img = dst[int(left_point[1]):int(heigth_point[1]),
                       int(left_point[0]):int(new_right_point[0]), ]
        card_imgs.append(card_img)
        #################################################
        # cv2.imshow("card", card_img)

    elif left_point[1] > right_point[1]:  # 负角度

        new_left_point = [left_point[0], heigth_point[1]]
        pts2 = np.float32([new_left_point, heigth_point,
                           right_point])  # 字符只是高度需要改变
        pts1 = np.float32([left_point, heigth_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
        point_limit(right_point)
        point_limit(heigth_point)
        point_limit(new_left_point)
        card_img = dst[int(right_point[1]):int(heigth_point[1]),
                       int(new_left_point[0]):int(right_point[0]), ]
        card_imgs.append(card_img)
        #################################################
        # cv2.imshow("card", card_img)
    return card_imgs


def confirmColor(card_imgs):
    # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        # 有转换失败的可能，原因来自于上面矫正矩形出错
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
        print("Analysing results:", "Blue:", blue, "Green:", green, "Yello:",
              yello, "Black:", black, "white:", white, "Card:", card_img_count)
        colors.append(color)
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
        card_imgs[card_index] = (card_img[yl:yh,
                                          xl:xr] if color != "green" or yl <
                                 (yh - yl) // 4 else
                                 card_img[yl - (yh - yl) // 4:yh, xl:xr])
        if need_accurate:  # 可能x或y方向未缩小，需要再试一次
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2,
                                            color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = (card_img[yl:yh,
                                          xl:xr] if color != "green" or yl <
                                 (yh - yl) // 4 else
                                 card_img[yl - (yh - yl) // 4:yh, xl:xr])
    return card_imgs, colors


def predictCard(card_imgs, colors):
    # 识别车牌中的字符
    predict_result = []
    roi = None
    card_color = None
    for i, color in enumerate(colors):
        if color in ("blue", "yello", "green"):
            card_img = card_imgs[i]
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 查找水平直方图波峰
            x_histogram = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peaks = find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                print("ERORR: peak of this card area less than 0:")
                continue
            # 认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            gray_img = gray_img[wave[0]:wave[1]]
            # 查找垂直直方图波峰
            row_num, col_num = gray_img.shape[:2]
            # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
            gray_img = gray_img[1:row_num - 1]
            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

            wave_peaks = find_waves(y_threshold, y_histogram)
            # 车牌字符数应大于6
            if len(wave_peaks) <= 6:
                print("ERORR: peak of this card area less than:",
                      len(wave_peaks))
                print("ERORR: it should be more than 6.")
                continue

            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            # 判断是否是左侧车牌边缘
            if (wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3
                    and wave_peaks[0][0] == 0):
                wave_peaks.pop(0)

            # 组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)

            # 去除车牌上的分隔点
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis / 3:
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

            if len(wave_peaks) <= 6:
                print("ERORR: peak of this card area less than:",
                      len(wave_peaks))
                print("ERORR: it should be more than 6.")
                continue

            part_cards = seperate_card(gray_img, wave_peaks)
            for i, part_card in enumerate(part_cards):
                # 可能是固定车牌的铆钉
                if np.mean(part_card) < 255 / 5:
                    print("Ignoring a point.")
                    continue
                part_card_old = part_card
                w = part_card.shape[1] // 3
                part_card = cv2.copyMakeBorder(part_card,
                                               0,
                                               0,
                                               w,
                                               w,
                                               cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
                part_card = cv2.resize(part_card, (SZ, SZ),
                                       interpolation=cv2.INTER_AREA)
                # cv2.imshow("part", part_card_old)
                # cv2.waitKey(0)
                part_card = preprocess_hog([part_card])
                # 读取模型
                model = SVM(C=1, gamma=0.5)
                model.load("svm.dat")
                modelchinese = SVM(C=1, gamma=0.5)
                modelchinese.load("svmchinese.dat")
                # 开始预测
                if i == 0:
                    resp = modelchinese.predict(part_card)
                    charactor = provinces[int(resp[0]) - PROVINCE_START]
                else:
                    resp = model.predict(part_card)
                    charactor = chr(int(resp[0]))
                # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                if charactor == "1" and i == len(part_cards) - 1:
                    if part_card_old.shape[0] / part_card_old.shape[
                            1] >= 8:  # 1太细，认为是边缘
                        print(part_card_old.shape)
                        continue
                predict_result.append(charactor)
            roi = card_img
            card_color = color
            break
    return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色


def predict(car_pic, resize_rate=1):
    # 预处理
    car_contours, pic_hight, pic_width, oldimg = preprocess(
        car_pic, resize_rate)
    # 定位车牌位置 并进行角度矫正
    card_imgs = localization(car_contours, pic_hight, pic_width, oldimg)
    # 识别车牌颜色 并进一步精确范围
    card_imgs, colors = confirmColor(card_imgs)
    # 识别车牌字符
    predict_result, roi, card_color = predictCard(card_imgs, colors)
    print(predict_result)
    print("Card color is:", card_color)
    cv2.imshow("Final Card", roi)

    return predict_result


def imreadex(filename):
    img_bgr = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    cv2.imshow("Source image", img_bgr)
    return img_bgr


if __name__ == '__main__':
    filename = "./car3.jpg"
    img_bgr = imreadex(filename)
    resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
    for resize_rate in resize_rates:
        print("resize_rate:", resize_rate)
        r = predict(img_bgr, resize_rate)
        if r:
            break
    cv2.waitKey(0)
