import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from itertools import chain
# import math
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import time

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True :

    success, image = capture.read()
    # image = cv2.imread('Dense Net\hand.jpg')
    hands, image = detector.findHands(image, draw=True)

    if hands :
        bbox = hands[0]['bbox']

        h1 = bbox[1] - 20
        h2 = bbox[1] + bbox[3] + 20
        w1 = bbox[0] - 20
        w2 = bbox[0] + bbox[2] + 20
        height = h2 - h1
        width = w2 - w1

        # print(f'w1 : {w1}, h1 : {h1}, height : {height}, width : {width}')

        # Preparing Coords Data
        coords = hands[0]['lmList']
        coords_xy = [[x,y] for [x,y,z] in coords]
        coords_xy = list(chain.from_iterable(coords_xy))
        # print(coords_xy)
        for i,coord in enumerate(coords_xy):
            if i%2 == 0:
                coords_xy[i] -= w1
            else :
                coords_xy[i] -= h1
            coords_xy[i] /= min(height,width)
        # print('New coords :', coords_xy)


    # img_crop = image[ h1:h2,w1:w2 ]

    cv2.imshow('image', image)
    # cv2.imshow('cropped',img_crop)
    cv2.waitKey(1)

