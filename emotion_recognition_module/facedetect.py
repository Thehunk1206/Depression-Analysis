import cv2

import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import time

import numpy as np

face_cascade = cv2.CascadeClassifier()
isLoaded= face_cascade.load('../cascade/haarcascade_frontalface_alt.xml')
print(isLoaded)

model = load_model('emotion_recogtion.h5')

classes = ['Angry','Happy','Neutral','Sad','Surprise']


cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    flipImage = cv2.flip(frame,1)
    grayFrame = cv2.cvtColor(flipImage,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(40,40)
        )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(flipImage, (x, y), (x+w, y+h), (0, 255,0), 1)
        roi_gray = grayFrame[y:y+w,x:x+h]
        roi_gray = cv2.resize(roi_gray,(48,48))
        img = image.img_to_array(roi_gray)
        img = np.expand_dims(img,axis=0)
        img/=255

        pred = model.predict(img)
        emotion = classes[np.argmax(pred[0])]
        cv2.putText(flipImage,emotion,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        print(emotion)




    #print("Detected {0} faces!".format(len(faces)))
    cv2.imshow("face",flipImage)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
del(model)
    
