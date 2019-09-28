import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}  #inverting the labels

cap = cv2.VideoCapture(0)

#Id = input('enter your id : ')

while True:
    #capture frame-by-frame
    _ ,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+h]
        #roi_color = frame[y:y+h, x:x+h]

        #recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 or conf <= 85:
            print([id_])
            print(labels[id_])
            '''font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels(id_)
            color = (255, 255, 255)
            cv2.putText(frame, name, (x, y), font, 1, 2, color)'''

        #cv2.imwrite("data/user." + Id + '.' + ".jpg", gray[y:y + h, x:x + w])
        img_item = "my_image1.png"
        cv2.imwrite(img_item,roi_gray)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)


    #display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

