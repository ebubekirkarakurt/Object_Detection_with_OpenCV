import cv2

img = cv2.imread(r'..\assets\images\face.png')
face_cascade = cv2.CascadeClassifier(r'..\assets\haar_cascades\frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'..\assets\haar_cascades\eye_cascade.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.4, 6)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

img2 = img[y:y + h, x:x + w]
gray2 = gray[y:y + h, x:x + w]

eyes = eye_cascade.detectMultiScale(gray2)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(img2, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
