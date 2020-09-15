import cv2 
#Load the cascade
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml') 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
#Load the image
img=cv2.imread('vk.jpg')
# #Convert image to gray scale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #Detect face
faces=face_cascade.detectMultiScale(gray,1.1,4)

# Detecting smile
smiles=smile_cascade.detectMultiScale(img,1.8,20)
#Detecting eyes
eye = eye_cascade.detectMultiScale(img, 1.2, 18)
# #Draw rectangle around the faces
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#Draw Rectangle around smiling face and each eye
for sx,sy,sw,sh in smiles:
    cv2.rectangle(img, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
for (x_eye, y_eye, w_eye, h_eye) in eye:
    cv2.rectangle(img,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2)
#Display
cv2.imshow('Detected Smile',img)
#Press any key to escape
cv2.waitKey()