import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import temp
import recognition
import os
import locatePlate

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
MIN_AREA = 500  # 车牌区域允许最大面积
PROVINCE_START = 1000

""" 
放缩图片大小
"""


def resize_pic(img_copy, resize_rate=1):
    pic_height, pic_width = img_copy.shape[:2]
    # 判断图片大小，放缩图片
    if pic_width > MAX_WIDTH:
        change_rate = MAX_WIDTH / pic_width
        img_copy = cv2.resize(imgSrc, (MAX_WIDTH, int(pic_height * change_rate)), interpolation=cv2.INTER_AREA)
        pic_height, pic_width = img_copy.shape[:2]
    # 根据定位车牌效果，放缩图片
    if resize_rate != 1:
        img_copy = cv2.resize(img_copy, (int(pic_width * resize_rate), int(pic_height * resize_rate)), interpolation=cv2.INTER_LANCZOS4)
        pic_height, pic_width = img_copy.shape[:2]
    # 尽量放大图片，放大车牌
    if pic_width < MAX_WIDTH:
        change_rate = MAX_WIDTH / pic_width
        img_copy = cv2.resize(imgSrc, (MAX_WIDTH, int(pic_height * change_rate)), interpolation=cv2.INTER_AREA)
        pic_height, pic_width = img_copy.shape[:2]

    return img_copy, pic_height, pic_width

def stretching(img_gray):
    """ 灰度拉伸 """
    maxi = float(img_gray.max())
    mini = float(img_gray.min())

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            img_gray[i, j] = 255 / (maxi - mini) * img_gray[i, j] - (255 * mini) / (maxi - mini)
    img_stretched = img_gray
    return img_stretched

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


""" 
颜色识别，主程序
"""


def color_recog_main(res_yellow, res_blue, res_green,oldimg, pic_width, pic_height):
    res = []
    res_name = []
    res.extend((res_blue, res_yellow, res_green))
    res_name.extend(("blue", "yellow", "green"))
    # 根据阈值找到对应颜色
    car_plates_all = []
    plate_colors_all = []
    predict_results=[]
    for res_id, res_hsv in enumerate(res):
        # 用颜色定位车牌
        car_plates, plate_colors = locatePlate.color_locate1(res_id, res_hsv, res_name, oldimg, pic_width, pic_height)
        car_plates_all.extend(car_plates)
        plate_colors_all.extend(plate_colors)
        if len(plate_colors)==1:
            break
        


    for i, plate_color in enumerate(plate_colors_all):
        # 做一次锐化处理
        car_plates_all[i] = cv2.resize(car_plates_all[i], (car_plates_all[i].shape[:2][1] * 2, car_plates_all[i].shape[:2][0] * 2), interpolation=cv2.INTER_AREA)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        card_img = cv2.filter2D(car_plates_all[i], -1, kernel=kernel)
        # cv2.imshow("car_plates_all" + str(i), car_plates_all[i])
        # 去除车牌铆钉，波峰分割
        plate_gray_img = cv2.cvtColor(car_plates_all[i], cv2.COLOR_BGR2GRAY)
        plate_gray_img=stretching(plate_gray_img)
        
        if plate_color == "green" or plate_color == "yellow":
            plate_gray_img = cv2.bitwise_not(plate_gray_img)
        ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        part_cards = temp.split_char3(plate_gray_img)

        # 字符识别
        predict_result = trainSvm.recog_card(part_cards)
        # print(i, " ", predict_result, " ", plate_color)
        # if len(predict_result)>=3:
        #     break
        predict_results.append(predict_result)
    return predict_results


""" 
边缘检测识别，主程序
"""


def edge_recog_main(res_all,oldimg, pic_width, pic_height):
    # 用颜色定位车牌
    car_plates, plate_colors = locatePlate.edge_locate1(res_all, oldimg, pic_width, pic_height)

    car_plates_all=car_plates
    plate_colors_all=plate_colors
    predict_results=[]
    for i, plate_color in enumerate(plate_colors_all):
        car_plates_all[i] = cv2.resize(car_plates_all[i], (car_plates_all[i].shape[:2][1] * 2, car_plates_all[i].shape[:2][0] * 2), interpolation=cv2.INTER_AREA)
        # 做一次锐化处理
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        card_img = cv2.filter2D(car_plates_all[i], -1, kernel=kernel)
        
        # cv2.imshow("car_plates_all" + str(i), car_plates_all[i])
        
        # 去除车牌铆钉，波峰分割
        plate_gray_img = cv2.cvtColor(car_plates_all[i], cv2.COLOR_BGR2GRAY)
        plate_gray_img=stretching(plate_gray_img)
        
        if plate_color == "green" or plate_color == "yellow":
            plate_gray_img = cv2.bitwise_not(plate_gray_img)
        ret, plate_gray_img = cv2.threshold(plate_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plate_gray_img = remove_upanddown_border(plate_gray_img)
        part_cards = temp.split_char3(plate_gray_img)

        # 字符识别
        predict_result = trainSvm.recog_card(part_cards)
        print(i, " ", predict_result, " ", plate_color)
        # if len(predict_result)>=3:
        #     break
        predict_results.append(predict_result)
    return predict_results

if __name__ == "__main__":
    trainSvm = recognition.TRAINSVM()
    trainSvm.train_svm()
    label_name=list("苏E2Y256")
    filename = "D:\\1MyLearningData\\LPR\hwtg\\carplate\\train\\"+ "".join(str(i) for i in label_name)+".jpg"
    # 根据路径读取图片
    imgSrc = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 获取输入图像的高和宽
    pic_height, pic_width = imgSrc.shape[:2]
    img_copy = imgSrc.copy()
    # 调整图片大小
    img_copy, pic_height, pic_width = resize_pic(img_copy, resize_rate=1)
    print(pic_height, pic_width)

    oldimg = img_copy.copy()

    # # 设置HSV颜色定位的up and down的阈值
    # colors = [([15, 55, 55], [50, 255, 255]),  # 黄色
    #           ([100, 43, 46], [124, 255, 255]),  # 蓝色
    #           ([0, 43, 116], [76, 211, 255])]  # 绿色
    
    colors = [([15, 43, 46], [50, 255, 255]),  # 黄色
            ([95, 43, 40], [130, 255, 255]),  # 蓝色
            ([35, 43, 46], [77, 255, 255])]  # 绿色

    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    # 设置掩码
    mask_yellow = cv2.inRange(hsv, np.array(colors[0][0]), np.array(colors[0][1]))
    mask_blue = cv2.inRange(hsv, np.array(colors[1][0]), np.array(colors[1][1]))
    mask_green = cv2.inRange(hsv, np.array(colors[2][0]), np.array(colors[2][1]))

    # 提取图片黄色、蓝色、绿色对应部分。这里直接全部全部提取，放在后面识别判断正确的车牌。
    res_yellow = cv2.bitwise_and(hsv, hsv, mask=mask_yellow)
    res_blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    res_green = cv2.bitwise_and(hsv, hsv, mask=mask_green)
    res_all = cv2.bitwise_and(hsv, hsv, mask=mask_yellow + mask_blue + mask_green)
    flag=1
    #颜色和边缘检测识别主程序
    predict_results1=color_recog_main(res_yellow, res_blue, res_green,oldimg, pic_width, pic_height)
    print("颜色结果",predict_results1)
    for result in predict_results1:
        if result==label_name:
            print("正确：",result)
            flag=0
            break  
    
    if flag: 
        predict_results2=edge_recog_main(img_copy,oldimg, pic_width, pic_height)
        print("边缘检测结果",predict_results2)
        for result in predict_results2:
            if result==label_name:
                print("正确：",result)
                break
            

    cv2.waitKey(0)
    cv2.destroyAllWindows()
