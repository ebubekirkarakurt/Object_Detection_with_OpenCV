import cv2
import imutils

cap = cv2.VideoCapture("video.mp4")

while 1:
    ret, frame = cap.read()

    frame = imutils.resize(frame, 400)

    if not ret:
        break

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (coordinate, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(1, 4), scale=1)

    for (x, y, w, h) in coordinate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
