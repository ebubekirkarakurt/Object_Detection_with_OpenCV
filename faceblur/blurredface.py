import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(r'..\assets\haar_cascades\frontalface_default.xml')
ksize = (50, 50)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    faces = face_cascade.detectMultiScale(frame, 1.1, 10)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        crop = frame[x:x + w, y:y + h]

        # cv2.imshow("Cropped", crop)
        # print(crop.shape)

        imgBlur = cv2.blur(crop, ksize)
        frame[y:y + h, x:x+w] = imgBlur

    cv2.imshow("Blurred Face", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
