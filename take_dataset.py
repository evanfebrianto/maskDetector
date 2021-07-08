import cv2
from os import path, makedirs

WIDTH, HEIGHT = 640, 360
OUTPUT_DIR = 'webcam_dataset'
class_names = ['incorrect_mask', 'mask', 'no_mask']

cap = cv2.VideoCapture(2)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

for class_name in class_names:
    if not path.exists(path.join(OUTPUT_DIR,class_name)):
        makedirs(path.join(OUTPUT_DIR,class_name))

state = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    print(class_names[state])
    cv2.imshow('Frame', image)
    k = cv2.waitKey(33)
    if k == ord('q'):
        break
    elif k == ord('1'):
        state = 0
    elif k == ord('2'):
        state = 1
    elif k == ord('3'):
        state = 2
cap.release()
cv2.destroyAllWindows() 