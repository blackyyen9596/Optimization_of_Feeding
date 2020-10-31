import numpy as np
from cv2 import cv2
import math
from matplotlib import pyplot as plt


def main():
    image = cv2.imread('image/very_large.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = get_h(hsv)
    # resize
    image = cv2.resize(image, (300, 400))
    hsv = cv2.resize(hsv, (300, 400))
    h = cv2.resize(h, (300, 400))
    # 顯示圖片
    cv2.imshow('input', image)
    cv2.imwrite('org.jpg', image)
    cv2.imshow('hsv', hsv)
    cv2.imwrite('hsv.jpg', hsv)
    cv2.imshow('h channel', h)
    # 從 h 獲取 green 通道
    g = get_channel(image, hsv, 35, 77)
    cv2.imshow('green', g)
    # 從 h 獲取 orange 通道
    o = get_channel(image, hsv, 11, 25)
    cv2.imshow('orange', o)
    cv2.imwrite('o.jpg', o)
    # 從 h 獲取 red 通道
    r = get_red_channel(image, hsv)
    cv2.imshow('red', r)
    # 從 h 獲取 yellow 通道
    y = get_channel(image, hsv, 26, 34)
    cv2.imshow('yellow', y)
    # 移動視窗位置
    cv2.moveWindow('input', 100, 50)
    cv2.moveWindow('hsv', 450, 50)
    cv2.moveWindow('h channel', 800, 50)
    cv2.moveWindow('green', 100, 500)
    cv2.moveWindow('orange', 450, 500)
    cv2.moveWindow('red', 800, 500)
    cv2.moveWindow('yellow', 1150, 500)

    # 計算輸入圖有多少像素
    totalPixel = get_NumOfTotalPixel(image)
    print("TotalPixel = ", totalPixel)

    # 轉至HSV後，各通道有值位置占整體比例(green,orange,red,yellow)
    ratio_g = ratio(image, totalPixel, g)
    ratio_o = ratio(image, totalPixel, o)
    ratio_r = ratio(image, totalPixel, r)
    ratio_y = ratio(image, totalPixel, y)
    print('ratio_green = %d percentage ' % (ratio_g))
    print('ratio_orange = %d percentage ' % (ratio_o))
    print('ratio_red = %d percentage ' % (ratio_r))
    print('ratio_yellow = %d percentage ' % (ratio_y))
    # cv2.imshow('green', ratio_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 獲取直方圖
    # hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    # 繪出直方圖
    # plt.figure()
    # plt.title('')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.plot(hist)
    # plt.xlim([0, 256])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def get_h(img):
    Img = img[:, :, 0]
    return Img


def get_channel(image, hsv, hmin, hmax):
    lower = np.array([hmin, 43, 46])
    upper = np.array([hmax, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


def get_red_channel(image, hsv):
    lower_1 = np.array([0, 43, 46])
    upper_1 = np.array([10, 255, 255])
    lower_2 = np.array([156, 43, 46])
    upper_2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_1, upper_1)
    mask2 = cv2.inRange(hsv, lower_2, upper_2)
    mask = mask1 + mask2
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


def get_NumOfTotalPixel(image):
    # height(rows) of image
    image_height = image.shape[0]
    print('height=', image_height)
    # width(colums) of image
    image_width = image.shape[1]
    print('width=', image_width)

    # scan all the image
    TotalPixel = 0
    for width in range(image_width):
        for height in range(image_height):
            if((image[height, width, 0] >= 225) & (image[height, width, 1] >= 225) & (image[height, width, 2] >= 225)):
                image[height, width, 0] = 255
                image[height, width, 1] = 255
                image[height, width, 2] = 255
            if((image[height, width, 0] != 255) & (image[height, width, 1] != 255) & (image[height, width, 2] != 255)):
                TotalPixel += 1
    print("TotalPixel=", TotalPixel)
    return TotalPixel


def ratio(image, totalPixel, color_series):
    image_height = image.shape[0]  # height(rows) of image
    image_width = image.shape[1]  # width(colums) of image
    # scan all the image
    cal = []
    colorPixel = 0
    for width in range(image_width):
        for height in range(image_height):
            if((color_series[height, width, 0] <= 10) & (color_series[height, width, 1] <= 10) & (color_series[height, width, 2] <= 10)):
                color_series[height, width, 0] = 0
                color_series[height, width, 1] = 0
                color_series[height, width, 2] = 0
            if((color_series[height, width, 0] != 0) & (color_series[height, width, 1] != 0) & (color_series[height, width, 2] != 0)):
                colorPixel += 1
    # 計算比率
    cal = (colorPixel/totalPixel)*100

    return cal


if __name__ == '__main__':
    main()
