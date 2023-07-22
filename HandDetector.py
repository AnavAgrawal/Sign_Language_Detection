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

IMG_SIZE = 28

custom_folder = 'Custom Data'

# Prediction stuff

model = load_model(r"Model Training\Models\No_MaxPool_787_897.h5")
# model.summary()

prediction_list = []  # To store the predictions for each frame
start_time = time.time()
interval = 1  # 1 second interval for printing the most common prediction


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

            # Preps image for model prediction
            imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            img_model = np.expand_dims(imgWhite, -1)
            img_model = np.expand_dims(img_model, 0)

            # Does the prediction
            preds = model.predict(img_model,verbose=0)
            prednum = np.argmax(preds)
            char_pred = chr(prednum+97)
            prediction_list.append(char_pred)
            # print('Predicted letter is ', char_pred)

            cv2.putText(imgWhite, char_pred, (100,100), cv2.FONT_HERSHEY_COMPLEX,30, color = (255,0,0), 
                        thickness=2)

            cv2.imshow('Resized_Window', imgWhite)

            # Prints most common prediction every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= interval:
                if prediction_list:
                    # Find the most common prediction
                    most_common_prediction = max(set(prediction_list), key=prediction_list.count)
                    print('Most frequent prediction:', most_common_prediction)
                    prediction_list.clear()  # Clear the list for the next second
                start_time = time.time()  # Reset the start time

            # if cv2.waitKey(20) & 0xFF == ord('d') :
            #     save_image = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            #     print('registered key press')
            #     cv2.imwrite('testimage.jpg', save_image)

            key = cv2.waitKey(1)
            if key == ord('r') :
                counter[0] += 1
                cv2.imwrite(f'{custom_folder}/Image_{counter[0]}.jpg', imgWhite)

    except Exception as e :
        print('Error bypassed : ', str(e))
        continue

    cv2.imshow('Image', image)
    cv2.waitKey(1)



