import cv2

# cap = cv2.VideoCapture(r'your path')  // find eyes in mp4
cap = cv2.VideoCapture(0)  # you can use your webcam

face_cascade = cv2.CascadeClassifier(r'..\assets\haar_cascades\frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'..\assets\haar_cascades\eye_cascade.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    frame2 = frame[y:y + h, x:x + w]
    gray2 = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(gray2)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame2, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    cv2.imshow("webcam", frame)
