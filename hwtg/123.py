import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import temp
import recognition
import os
def color_oc_operation(hsv_gray):
    # 1.灰度图的高斯平滑去噪
    gray_gauss = cv2.GaussianBlur(hsv_gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

    # 2.开运算，与上次运算结果融合
    kernel = np.ones((5, 5), np.uint8)
    img_opening = cv2.morphologyEx(gray_gauss, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_gauss, 1, img_opening, -1, 0)


    # 3.二值化，并Canny边缘化(或不边缘)
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = img_thresh

    # 4.使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    
    
def color_oc_operation(hsv_gray):
    # 1.灰度图的高斯平滑去噪
    gray_gauss = cv2.GaussianBlur(hsv_gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

    # 2.开运算，与上次运算结果融合
    kernel = np.ones((3, 3), np.uint8)
    img_opening = cv2.morphologyEx(gray_gauss, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 25), np.uint8)
    img_opening = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)

    # 3.二值化，并Canny边缘化(或不边缘)
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = img_thresh

    # 4.使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def color_oc_operation(hsv_gray):
    # 1.灰度图的高斯平滑去噪
    gray_gauss = cv2.GaussianBlur(hsv_gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

    # 2.开运算，与上次运算结果融合
    kernel = np.ones((5, 15), np.uint8)
    img_opening = cv2.morphologyEx(gray_gauss, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_gauss, 1, img_opening, -1, 0)


    # 3.二值化，并Canny边缘化(或不边缘)
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_edge = cv2.Canny(img_thresh, 100, 200)
    img_edge=img_thresh

    # 4.使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 14), np.uint8)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    # 5.查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours