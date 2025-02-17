import cv2
import numpy 
import pandas as pd
import gdown
from keras.models import model_from_json

# Google Drive file ID extracted from the shared link
file_id = "15wk718aKgu9si2aX41xGYJhhIWFScJaD"
url = f"https://drive.google.com/uc?id={file_id}"
output = "emotiondetector.h5"

# Download the file
gdown.download(url, output, quiet=False)
print("Model downloaded successfully!")

with open("emotiondetector.json", "r") as file:
    model_json = file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract(image):
    F = numpy.array(image)
    F = F.reshape(1,48,48,1)
    return F/255.0

webcam=cv2.VideoCapture(0)
face_labels = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract(image)
            pred = model.predict(img)
            prediction_label = face_labels[pred.argmax()]
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",im)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
