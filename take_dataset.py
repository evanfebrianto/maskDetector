import cv2
import mediapipe as mp
import time
from os import path, makedirs

PADDING_SCALE = 0.1
WIDTH, HEIGHT = 640, 360
OUTPUT_DIR = 'webcam_dataset'
class_names = ['incorrect_mask', 'mask', 'no_mask']

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(2)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

for class_name in class_names:
    if not path.exists(path.join(OUTPUT_DIR,class_name)):
        makedirs(path.join(OUTPUT_DIR,class_name))

state = 0
save_img = False
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.2) as face_mesh:
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
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        original_frame_bgr = image.copy()
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                h, w, c = image.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy

                dx, dy = (cx_max-cx_min, cy_max-cy_min)

                # Add padding to draw BB
                cx_min -= int(dx * PADDING_SCALE)
                cy_min -= int(dy * PADDING_SCALE)
                cx_max += int(dx * PADDING_SCALE)
                cy_max += int(dy * PADDING_SCALE)
                cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (0, 0, 255), 2)
                
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
                    fontScale=0.8, color=(0, 50, 255), thickness=2)
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
        elif k == ord('s'):
            save_img = True
cap.release()
cv2.destroyAllWindows() 