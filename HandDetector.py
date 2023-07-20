import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# For bigger image window
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resized_Window", 400, 400)

IMG_SIZE = 28

while True :

    success, image = capture.read()
    hands = detector.findHands(image, draw=False)

    try :
        if hands:

            bbox = hands[0]['bbox']

            # Draws the rectangle
            cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20), 
                                (bbox[0] + bbox[2] + 20, 
                                bbox[1] + bbox[3] + 20),
                                        (255, 0, 255), 2)
            
            # Creates a blank image
            imgWhite = np.ones((IMG_SIZE, IMG_SIZE,3), dtype=np.uint8)*255

            # Crops the image
            onlyhand = image[bbox[1] - 20 : bbox[1] + bbox[3] + 20, bbox[0] - 20:bbox[0] + bbox[2] + 20]

            onlyhandShape = onlyhand.shape
            height = onlyhandShape[0]
            width = onlyhandShape[1]

            cv2.imshow('Only hand', onlyhand)

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

    except :
        print('Error bypassed')
        continue

    cv2.imshow('Image', image)

    cv2.waitKey(1)