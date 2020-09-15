import cv2
#Load the cascade
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml') 
#capture video
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
def detect(gray,frame):
    #Face Detection
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    for x,y,w,h in faces:
        #Draw Rectangle around each face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #Simply cropping original and grayscale image for better result
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        #Detect smile
        smiles=smile_cascade.detectMultiScale(roi_gray,1.8,20)
        #Draw rectangle around smile
        for sx,sy,sw,sh in smiles:
                cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
        #Detect eyes
        eye = eye_cascade.detectMultiScale(roi_gray, 1.2, 18) 
        #Draw rectangle around each eye
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(roi_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
    return frame
while True:
    _,frame=cap.read()
    #convert original image to grayscale 
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    #Display
    cv2.imshow('video',canvas)
    #break if escape key is pressed
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
#release the video capturing object
cap.release()
cv2.destroyAllWindows()

