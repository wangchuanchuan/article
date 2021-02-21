# """"
# 文档二值化算法实现, 全部返还白底黑字（255\0）
# 动态二值化
# 循环背景差分二值化
# sauvola二值化
# 以下算法集成在skimage中，通过skimage_wrapper函数调用：
# isodata
# li
# mean
# minimum
# otsu
# triangle
# yen
# """
from skimage import restoration
from skimage import filters
from scipy.signal import convolve2d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns

def adaptive_binary(img, bi_filter_size=5):
# """
# opencv动态二值化， c校正为global值
# :param bi_filter_size: 双边滤波的大小
# :param img:numpy格式图像数据, 灰度图
# :return:二值化图像（0， 255）， 黑底白字
# """
# 保留轮廓的模糊算法
img = cv2.bilateralFilter(img, bi_filter_size, 20, 50)
# 计算C，利用k mean对图像灰度进行而分类，默认低灰度类为背景，取C=类中心
c = np.mean(img) * 0.05
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=5, C=c)
return img

def iteretive_binary(img, bi_filter_size=5):
# """
# 二值化实现, 循环减去背景平均，再动态二值化
# inference paper: A binarization algorithm specialized on document images and photos
# :param img: gray image
# :return:黑底白字
# """
# 双边
img = 255 - cv2.bilateralFilter(img, bi_filter_size, 20, 50)
gray_mean_prev = 1
gray_mean = 0
time_be = time.time()
while abs(gray_mean - gray_mean_prev) > 0.1:
# 当运行时间超过60秒，break
if (time.time() - time_be) > 60:
break
gray_mean_prev = gray_mean
gray_mean = (np.mean(img))
img = np.float32(img)
img = img - gray_mean
img[img < 0] = 0
img = cv2.equalizeHist(np.uint8(img))
img = adaptive_binary(255 - img)
return img

def otsu_binary(img, bi_filter_size=5):
# """
# 简单的OTSU二值化，适用于印刷扫描（无光照影响）的图像
# :param img:
# :return:
# """
img = cv2.bilateralFilter(img, bi_filter_size, 20, 50)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
return img

# Deprecated
def sauvola_obsolete(img, bi_filter_size=5, filter_shape=5):
# """
# （弃用， 计算时间太长，改用skimage集成）
# Sauvola's 二值化算法实现 (目前效果最好，性能最差)
# T = m * [1 + K*(s/R - 1)]; m，s为局部均值方差， k=0.5, R, 方差的动态范围，经验取方差范围某个比例点，而非论文所说128
# :param bi_filter_size: 双边滤波大小
# :param filter_shape: 局部操作大小
# :param img:
# :return:
# """
assert (filter_shape % 2) == 1, 'filter shape must be odd'
# 双边
img = cv2.bilateralFilter(img, bi_filter_size, 20, 50)
# 转换数据格式保留精度
img = np.float32(img)
# padding
padding_width = int((filter_shape - 1) / 2)
img_padding = np.lib.pad(img, ((padding_width, padding_width), (padding_width, padding_width)), 'edge')
# 计算方差
h, w = img.shape
img_mean = cv2.blur(img, (filter_shape, filter_shape))
img_s = np.zeros_like(img)
for h_step in range(h):
for w_step in range(w):
local_box = img_padding[h_step:h_step+filter_shape, w_step:w_step+filter_shape]
img_s[h_step, w_step] = np.std(local_box)
# 根据公式计算阈值
k = 0.1
# 方差的动态范围
r = np.max(img_s) / 3.0
img_threshold = img_mean * (1 + k*(img_s / r - 1))
img[img > img_threshold] = 0
img[img > 0] = 255
return np.uint8(img)

def sauvola(img, bi_filter_size=5, window_size=21, k=0.1, r=None):
# """
# sauvola的算法实现，基于skimage
# 主要参数：window_size和k影响最后结果；注意大图像window_size取大点，k越小文字越清晰，但是背景噪点越多
# :param img: 灰度图像
# :param bi_filter_size: 双边滤波的大小，大图像常用11，小图像5
# :param window_size: 局部滤波计算均值方差的窗口大小，小图像取小大图像取大
# :param k: 可理解为背景与前景的区别程度，越大背景噪点压制越好，但是文字也越淡
# :param r: 方差动态范围，默认不设
# :return:
# """
assert (window_size % 2) == 1, 'filter shape must be odd'
# 双边
img = cv2.bilateralFilter(img, bi_filter_size, 20, 50)
# 转换数据格式保留精度
threshold = filters.threshold_sauvola(img, window_size=window_size, k=k, r=r)
img = img > threshold
return np.uint8(img) * 255

def skimage_binary_wrapper(img, fn, bi_filter_size=5):
# """
# skimage中全局二值化封装函数；
# fn为二值化函数：
# kimage.filters.threshold_otsuReturn threshold value based on Otsu’s method.
# skimage.filters.threshold_yen
# skimage.filters.threshold_isodata
# skimage.filters.threshold_li
# skimage.filters.threshold_minimum
# skimage.filters.threshold_mean
# :param img: 灰度图像
# :param fn: skimage二值化函数
# :param bi_filter_size:
# :return:
# """
# 双边
img = cv2.bilateralFilter(img, bi_filter_size, 20, 50)
threshold = fn(img)
img = img > threshold
return img

