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
    print(color)
    colors.append(color)
    print(blue, green, yello, black, white, card_img_count)
    # cv2.imshow("color", card_img)
    # cv2.waitKey(0)
    if limit1 == 0:
        continue
    # 以上为确定车牌颜色
    # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
    xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
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
        else card_img[yl - (yh - yl) // 4 : yh, xl:xr]
    )
    if need_accurate:  # 可能x或y方向未缩小，需要再试一次
        card_img = card_imgs[card_index]
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
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
        else card_img[yl - (yh - yl) // 4 : yh, xl:xr]
    )
# 以上为车牌定位
# 以下为识别车牌中的字符
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
        ret, gray_img = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # 查找水平直方图波峰
        x_histogram = np.sum(gray_img, axis=1)
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        x_threshold = (x_min + x_average) / 2
        wave_peaks = find_waves(x_threshold, x_histogram)
        if len(wave_peaks) == 0:
            print("peak less 0:")
            continue
        # 认为水平方向，最大的波峰为车牌区域
        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        gray_img = gray_img[wave[0] : wave[1]]
        # 查找垂直直方图波峰
        row_num, col_num = gray_img.shape[:2]
        # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
        gray_img = gray_img[1 : row_num - 1]
        y_histogram = np.sum(gray_img, axis=0)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram) / y_histogram.shape[0]
        y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

        wave_peaks = find_waves(y_threshold, y_histogram)

        # for wave in wave_peaks:
        # 	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
        # 车牌字符数应大于6
        if len(wave_peaks) <= 6:
            print("peak less 1:", len(wave_peaks))
            continue

        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        max_wave_dis = wave[1] - wave[0]
        # 判断是否是左侧车牌边缘
        if (
            wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3
            and wave_peaks[0][0] == 0
        ):
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
            wave_peaks = wave_peaks[i + 1 :]
            wave_peaks.insert(0, wave)

        # 去除车牌上的分隔点
        point = wave_peaks[2]
        if point[1] - point[0] < max_wave_dis / 3:
            point_img = gray_img[:, point[0] : point[1]]
            if np.mean(point_img) < 255 / 5:
                wave_peaks.pop(2)

        if len(wave_peaks) <= 6:
            print("peak less 2:", len(wave_peaks))
            continue
        part_cards = seperate_card(gray_img, wave_peaks)
        for i, part_card in enumerate(part_cards):
            # 可能是固定车牌的铆钉
            if np.mean(part_card) < 255 / 5:
                print("a point")
                continue
            part_card_old = part_card
            # w = abs(part_card.shape[1] - SZ)//2
            w = part_card.shape[1] // 3
            part_card = cv2.copyMakeBorder(
                part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
            # cv2.imshow("part", part_card_old)
            # cv2.waitKey(0)
            # cv2.imwrite("u.jpg", part_card)
            # part_card = deskew(part_card)
            part_card = preprocess_hog([part_card])
            if i == 0:
                resp = self.modelchinese.predict(part_card)
                charactor = provinces[int(resp[0]) - PROVINCE_START]
            else:
                resp = self.model.predict(part_card)
                charactor = chr(resp[0])
            # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
            if charactor == "1" and i == len(part_cards) - 1:
                if part_card_old.shape[0] / part_card_old.shape[1] >= 8:  # 1太细，认为是边缘
                    print(part_card_old.shape)
                    continue
            predict_result.append(charactor)
        roi = card_img
        card_color = color
        break

return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色
