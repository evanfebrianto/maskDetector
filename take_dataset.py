import cv2

WIDTH, HEIGHT = 640, 360

cap = cv2.VideoCapture(2)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    cv2.imshow('Frame', image)
    if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows() 