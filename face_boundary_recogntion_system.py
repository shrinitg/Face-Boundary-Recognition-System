import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        
        img=cv2.putText(img, 'Face detected', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        
        roi = img[y-30:y+h+10, x:x+w]
        
        img_gray_blur = cv2.blur(gray,(10,10))
    
        canny_edge = cv2.Canny(img_gray_blur, 10, 70)
    
        ret, mask = cv2.threshold(canny_edge, 30, 255, cv2.THRESH_BINARY)
        
        mask[30:180, 40:100] = 0
        
        bitAnd = cv2.bitwise_and(mask,canny_edge)
        
        mask[50:100,]=0
        mask[300:500,]=0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            if cv2.arcLength(contour,True) >20:
                cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
        
        cv2.imshow('mask', mask)
        cv2.imshow('bitAnd', bitAnd)

    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()