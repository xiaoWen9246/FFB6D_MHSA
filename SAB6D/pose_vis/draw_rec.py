import os
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import cv2 as cv
from PIL import Image, ImageDraw

img_path = 'ground_truth/286.jpg'
img = Image.open(img_path)
a = ImageDraw.ImageDraw(img)
# rectangle 坐标的参数格式为左上角（x1, y1），右下角（x2, y2）。
a.rectangle(((205, 72), (375, 219)), fill=None, outline='yellow', width=2)
img.save('new_test_2.jpg')
