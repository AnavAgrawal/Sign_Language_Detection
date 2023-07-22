import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

counter = [0,0,0]

# For bigger image window
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resized_Window", 400, 400)

IMG_SIZE = 80

custom_folder = 'Custom Model Training\Data_v2'

# Prediction stuff

while True :

    success, image = capture.read()
    hands, img = detector.findHands(image, draw=True)

    try :
        if hands:

            bbox = hands[0]['bbox']

            # # Draws the rectangle
            # cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20), 
            #                     (bbox[0] + bbox[2] + 20, 
            #                     bbox[1] + bbox[3] + 20),
            #                             (255, 0, 255), 2)
            
            # Creates a blank image
            imgWhite = np.ones((IMG_SIZE, IMG_SIZE,3), dtype=np.uint8)*255

            # Crops the image
            onlyhand = image[bbox[1] - 20 : bbox[1] + bbox[3] + 20, bbox[0] - 20:bbox[0] + bbox[2] + 20]

            onlyhandShape = onlyhand.shape
            height = onlyhandShape[0]
            width = onlyhandShape[1]

            # cv2.imshow('Only hand', onlyhand)

            # To center the image
            heightOffset = 0
            widthOffset = 0

            # Resizes the image
            if height > width : 
                width = math.floor(width*IMG_SIZE/height)
                height = IMG_SIZE
                onlyhand = cv2.resize(onlyhand, (width,height))
                widthOffset = math.floor((IMG_SIZE - width)/2)

            else : 
                height = math.floor(height*IMG_SIZE/width)
                width = IMG_SIZE
                onlyhand = cv2.resize(onlyhand, (width,height))
                heightOffset = math.floor((IMG_SIZE - height)/2)

            # Adds the image onto the blank image
            imgWhite[heightOffset:height+heightOffset, widthOffset:width+widthOffset] = onlyhand

            cv2.imshow('Resized_Window', imgWhite)

            key = cv2.waitKey(1)
            if key == ord('r') :
                counter[0] += 1
                cv2.imwrite(f'{custom_folder}/Rock/Image_{counter[0]}.jpg', imgWhite)
                print('Rock Image Number ', counter[0], 'Saved')
            elif key == ord('p') :
                counter[1] += 1
                cv2.imwrite(f'{custom_folder}/Paper/Image_{counter[1]}.jpg', imgWhite)
                print('Paper Image Number ', counter[1], 'Saved')
            elif key == ord('s') :
                counter[2] += 1
                cv2.imwrite(f'{custom_folder}/Scissors/Image_{counter[2]}.jpg', imgWhite)
                print('Scissors Image Number ', counter[2], 'Saved')

    except Exception as e :
        print('Error bypassed : ', str(e))
        continue

    cv2.imshow('Image', image)
    cv2.waitKey(1)



