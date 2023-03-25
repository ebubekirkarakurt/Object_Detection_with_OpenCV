import cv2

# cap = cv2.VideoCapture('your path') // oynatmak istediğiniz videonun yolunu girin.
cap = cv2.VideoCapture(0)#you can use your webcam
face_cascade = cv2.CascadeClassifier(r'..\assets\haar_cascades\frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
