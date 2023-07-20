import cv2 as cv

# img = cv.imread('photo.png')
# cv.imshow('ss',img)

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# capture.set(3,28)
# capture.set(4,28)

while True : 
    isTrue, frame = capture.read()

    # canny = cv.Canny(frame, 125,175)

    faces_rect = haar_cascade.detectMultiScale(frame,scaleFactor=1.1, minNeighbors=5)

    for x,y,w,h in faces_rect :
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
        only_face = frame[y:y+h,x:x+w]
        only_face = cv.resize(only_face,(300,300))
        cv.imshow('Face', only_face)

    cv.imshow('Video', frame)
    # cv.imshow('Video', canny)

    if cv.waitKey(20) & 0xFF == ord('d') :
        break 

capture.release()
cv.destroyAllWindows()