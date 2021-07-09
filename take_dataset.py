import cv2
import mediapipe as mp
import time
from os import path, makedirs
from lib import helper

parser = helper.parse_config('config.ini')

PADDING_SCALE = parser['FLOAT']['PADDING']
WIDTH = parser['INT']['WIDTH']
HEIGHT = parser['INT']['HEIGHT']
SHORT_RANGE = parser['BOOLEAN']['SHORT_RANGE']
DETECTION_CONFIDENCE = parser['FLOAT']['DETECTION_CONFIDENCE']
COLOR_MASK = parser['TUPLE']['GREEN']
COLOR_INCORRECT = parser['TUPLE']['YELLOW']
COLOR_NO_MASK = parser['TUPLE']['RED']

OUTPUT_DIR = 'webcam_dataset'
class_names = ['incorrect_mask', 'mask', 'no_mask']


# For webcam input:
cap = cv2.VideoCapture(2)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

for class_name in class_names:
    if not path.exists(path.join(OUTPUT_DIR,class_name)):
        makedirs(path.join(OUTPUT_DIR,class_name))

state = 0
selected_color = COLOR_INCORRECT
save_img = False
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0 if SHORT_RANGE else 1, 
    min_detection_confidence=DETECTION_CONFIDENCE) as face_detection:
     while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        original_frame_bgr = image.copy()
        if results.detections:
            for detection in results.detections:
                cx_min, cy_min, cx_max, cy_max = helper.get_bb(image, detection)
                if cx_min is not None:
                    dx, dy = (cx_max-cx_min, cy_max-cy_min)

                    # Add padding to draw BB
                    cx_min -= int(dx * PADDING_SCALE)
                    cy_min -= int(dy * PADDING_SCALE)
                    cx_max += int(dx * PADDING_SCALE)
                    cy_max += int(dy * PADDING_SCALE)
                    cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), selected_color, 2)
                    
                    # Save face dataset
                    crop_img = original_frame_bgr[cy_min:cy_max, cx_min:cx_max]
                    if save_img:
                        image_path = path.join(OUTPUT_DIR, class_names[state],
                                        str(time.time())+'.jpg')
                        try:
                            cv2.imwrite(image_path, crop_img)
                        except: # ignore empty frame
                            continue
                        print('Saved: {}'.format(image_path))
                        save_img = False

        # print(class_names[state])
        cv2.putText(image, class_names[state], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.8, color=selected_color, thickness=2)
        cv2.imshow('Frame', image)
        k = cv2.waitKey(33)
        if k == ord('q'):
            break
        elif k == ord('1'):
            state = 0
            selected_color = COLOR_INCORRECT
        elif k == ord('2'):
            state = 1
            selected_color = COLOR_MASK
        elif k == ord('3'):
            state = 2
            selected_color = COLOR_NO_MASK
        elif k == ord('s'):
            save_img = True
cap.release()
cv2.destroyAllWindows() 