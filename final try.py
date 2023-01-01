from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import cv2

DIRECTORY = "DATA"
CATEGORIES = ["wearing mask", "not_wearing_a_mask"]
data = []
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    labels = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image=cv2.imread(img_path)
        image=cv2.resize(image,(224,224))
        data.append([image,labels])

x,y=[],[]
for i,j in data:
    x.append(i)
    y.append(j)
import numpy as np
x=np.array(x)
y=np.array(y)
x=x.reshape(1822,224*224*3)

LABELS= {0:'MASK DETECTED',1:'NO MASK DETECTED'}
(x_train, x_test, y_train, y_test) = train_test_split(x, y,
	test_size=0.20, stratify=y, random_state=42)
print(x_train.shape)

p=PCA(n_components=3)
x_train=p.fit_transform(x_train)
x_train[0]
x_test=p.fit_transform(x_test)
print("starting to fit")

#  create model instance
log= LogisticRegression()

#  Model Fitting
log = log.fit(x_train, y_train)

log_pred = log.predict(x_test)

log_accuracy=accuracy_score(y_test,log_pred)
print("accuracy score:",log_accuracy)
print("starting to detect")
import cv2
faceCascade=cv2.CascadeClassifier(r"C:\Users\santr\Downloads\haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)
data = []
while True:
    success, img = video_cap.read()
    if success:
        faces = faceCascade.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_image=img[y:y+h, x:x+w, :]
            face_image=cv2.resize(face_image,(224,224))
            face_image=face_image.reshape(1,-1)
            face_image = p.transform(face_image)
            pred = log.predict(face_image)
            l = LABELS[int(pred)]
            print(l)
        cv2.imshow("Resultant", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
video_cap.release()
