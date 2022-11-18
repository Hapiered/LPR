import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image


# 波峰
# image2二值化的图片
def split_char(image2):

    # 水平投影
    h1, w1 = image2.shape  # 返回高和宽
    image3 = image2.copy()
    a = [0 for z in range(0, h1)]  # 初始化一个长度为w的数组，用于记录每一行的黑点个数
    # 记录每一行的波峰
    for j in range(0, h1):
        for i in range(0, w1):
            if image3[j, i] == 0:
                a[j] += 1
                image3[j, i] = 255

    for j in range(0, h1):
        for i in range(0, a[j]):
            image3[j, i] = 0

    # plt.imshow(image3, cmap=plt.gray())  # 灰度图正确的表示方法
    # plt.show()
    cv2.imshow('image3', image3)

    # 垂直投影
    h2, w2 = image2.shape  # 返回高和宽
    image4 = image2.copy()
    b = [0 for z in range(0, w2)]  # b = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
    # 记录每一列的波峰
    for j in range(0, w2):  # 遍历一列
        for i in range(0, h2):  # 遍历一行
            if image4[i, j] == 0:  # 如果该点为黑点
                b[j] += 1  # 该列的计数器加一，最后统计出每一列的黑点个数
                image4[i, j] = 255  # 记录完后将其变为白色，相当于擦去原图黑色部分

    for j in range(0, w2):
        for i in range((h2 - b[j]), h2):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            image4[i, j] = 0  # 涂黑

    # plt.imshow(image4, cmap=plt.gray())
    # plt.show()
    cv2.imshow('image4', image4)

    # 分割字符
    Position = []
    start = 0
    a_Start = []
    a_End = []

    # 根据水平投影获取垂直分割位置
    for i in range(len(a)):
        if a[i] > 0 and start == 0:
            a_Start.append(i)
            start = 1
        if a[i] <= 0 and start == 1:
            a_End.append(i)
            start = 0
    # print(len(a_Start), len(a_End))

    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(min(len(a_Start), len(a_End))):
        # 获取行图像
        cropImg = image2[a_Start[i]:a_End[i], 0:w1]
        # 对行图像进行垂直投影
        bstart = 0
        bend = 0
        b_Start = 0
        b_End = 0
        for j in range(len(b)):
            if b[j] > 0 and bstart == 0:
                b_Start = j
                bstart = 1
                bend = 0
            if b[j] <= 0 and bstart == 1:
                b_End = j
                bstart = 0
                bend = 1
            if bend == 1:
                Position.append([b_Start, a_Start[i], b_End, a_End[i]])
                bend = 0
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # 将灰度图转为RGB彩图

    # 根据确定的位置分割字符
    for m in range(len(Position)):
        # 第一个参数是原图；第二个参数是矩阵的左上点坐标；第三个参数是矩阵的右下点坐标；第四个参数是画线对应的rgb颜色；第五个参数是所画的线的宽度
        cv2.rectangle(image2, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 0, 255), 5)
    print("================")
    cv2.imshow('rect', image2)


# 投影法分割字符
# 计算每一列的白色像素块作为分割区域
def split_char2(img_thre):
    # 4、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    # 分割图像
    ratio = 0.85

    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (ratio * black_max if arg else ratio * white_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_

    n = 1
    start = 1
    end = 2
    word = []
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > ((1 - ratio) * white_max if arg else (1 - ratio) * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = img_thre[1:height, start:end]
                cj = cv2.resize(cj, (15, 30))
                word.append(cj)
    return word




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

#波峰
def split_char3(gray_img):
    # 查找水平直方图波峰
    x_histogram = np.sum(gray_img, axis=1)
    x_min = np.min(x_histogram)
    x_average = np.sum(x_histogram) / x_histogram.shape[0]
    x_threshold = (x_min + x_average) / 2
    wave_peaks = find_waves(x_threshold, x_histogram)
    if len(wave_peaks) == 0:
        # print("peak less 0:")
        return []
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

    # for wave in wave_peaks:
    # cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
    # 车牌字符数应大于6
    if len(wave_peaks) <= 6:
        # print("peak less 1:", len(wave_peaks))
        return []

    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
    max_wave_dis = wave[1] - wave[0]
    # 判断是否是左侧车牌边缘
    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
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
    if len(wave_peaks)<=2:
        return []
    point = wave_peaks[2]
    if point[1] - point[0] < max_wave_dis / 3:
        point_img = gray_img[:, point[0]:point[1]]
        if np.mean(point_img) < 255 / 5:
            wave_peaks.pop(2)

    if len(wave_peaks) <= 6:
        # print("peak less 2:", len(wave_peaks))
        return []
    part_cards = seperate_card(gray_img, wave_peaks)
    # for i, char in enumerate(part_cards):
    #     cv2.imshow("character" + str(i), char)
    return part_cards


def split_char4(img_gray):
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(img_gray, kernel)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    words = sorted(words,key=lambda s:s[0],reverse=False)
    i = 0
    for word in words:
        # 根据轮廓的外接矩形筛选轮廓
        if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
            i = i+1
            splite_image = img_gray[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            splite_image = cv2.resize(splite_image,(25,40))
            word_images.append(splite_image)

    for i,j in enumerate(word_images):
        plt.subplot(1,len(word_images),i+1)
        plt.imshow(word_images[i],cmap='gray')
    plt.show()
    return word_images