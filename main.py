from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import mediapipe as mp
from lib import helper
import time, datetime
import numpy as np
from threading import Thread
from multiprocessing import Process
import lib.emailClass
import os

parser = helper.parse_config('config.ini')

SHORT_RANGE = parser['BOOLEAN']['SHORT_RANGE']
DEBUG = parser['BOOLEAN']['DEBUG']
SAVE_VIDEO = parser['BOOLEAN']['SAVE_VIDEO']
SENDING_EMAIL = parser['BOOLEAN']['SENDING_EMAIL']

WIDTH = parser['INT']['WIDTH']
HEIGHT = parser['INT']['HEIGHT']
CAM_ID = parser['INT']['CAMERA_ID']
EMAIL_INTERVAL = parser['INT']['INTERVAL_SECOND']
DETECTION_BUFFER = parser['INT']['DETECTION_BUFFER']

LOG_DIR = parser['STRING']['LOG_DIR']

PADDING_SCALE = parser['FLOAT']['PADDING']
DETECTION_CONFIDENCE = parser['FLOAT']['DETECTION_CONFIDENCE']

MEAN = parser['LIST']['MEAN']
STD = parser['LIST']['STD']

COLOR_MASK = parser['TUPLE']['GREEN']
COLOR_INCORRECT = parser['TUPLE']['YELLOW']
COLOR_NO_MASK = parser['TUPLE']['RED']
COLOR_INVALID = parser['TUPLE']['BLACK']


print('\n{} {} {}'.format('*'*16, 'LOADED PARAMETER','*'*16))
print('SHORT_RANGE: {}'.format(SHORT_RANGE))
print('DEBUG: {}'.format(DEBUG))
print('SAVE_VIDEO: {}'.format(SAVE_VIDEO))
print('SENDING_EMAIL: {}\n'.format(SENDING_EMAIL))
print('WIDTH: {}'.format(WIDTH))
print('HEIGHT: {}'.format(HEIGHT))
print('CAM_ID: {}'.format(CAM_ID))
print('EMAIL_INTERVAL: {}'.format(EMAIL_INTERVAL))
print('DETECTION_BUFFER: {}\n'.format(DETECTION_BUFFER))
print('LOG_DIR: {}\n'.format(LOG_DIR))
print('PADDING_SCALE: {}'.format(PADDING_SCALE))
print('DETECTION_CONFIDENCE: {}\n'.format(DETECTION_CONFIDENCE))
print('MEAN: {}'.format(MEAN))
print('STD: {}\n'.format(STD))
print('COLOR_MASK: {}'.format(COLOR_MASK))
print('COLOR_INCORRECT: {}'.format(COLOR_INCORRECT))
print('COLOR_NO_MASK: {}'.format(COLOR_NO_MASK))
print('COLOR_INVALID: {}'.format(COLOR_INVALID))
print('{}\n'.format('*'*50))


def sendingEmail():
    global isSendingEmail
    if isSendingEmail:
        time_now = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
        email.setProperties(subject='No Mask Report at {}'.format(time_now),
            body_message='Hi, please kindly check the attached file for your reference.')
        email.sendEmail()
        isSendingEmail = False

if __name__ == "__main__":
    # For webcam input:
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    transformer = transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    class_names = ['incorrect_mask', 'mask', 'no_mask']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = models.resnet18(pretrained=True)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))  # make the change
    model.load_state_dict(torch.load('models/model_ft.pth'))
    model = model.to(device)
    model.eval()

    mp_face_detection = mp.solutions.face_detection

    if SAVE_VIDEO:
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (WIDTH,HEIGHT))

    # initialize some variables
    procs = []
    FPS_data = []
    people_counter = 0
    counter_noMask, counter_incorrect = 0, 0
    email_time = time.time()
    current_dir = str(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

    email = lib.emailClass.EmailModule(configFile='config.ini')
    isSendingEmail = False

    with mp_face_detection.FaceDetection(
        model_selection=0 if SHORT_RANGE else 1, 
        min_detection_confidence=DETECTION_CONFIDENCE) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # Start timer
            start_time = time.time()
            
            # Initialize bool value to determine label image
            isNoMask, isIncorrect = False, False

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            original_frame_rgb = image.copy()
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_detection.process(image)


            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Reset people_counter
            people_counter = 0

            if results.detections:
                for detection in results.detections:
                    cx_min, cy_min, cx_max, cy_max = helper.get_bb(image, detection)
                    if cx_min is not None:
                        people_counter += 1
                        dx, dy = (cx_max-cx_min, cy_max-cy_min)

                        # Add padding to classify image
                        if not DEBUG:
                            cx_min -= int(dx * PADDING_SCALE)
                            cy_min -= int(dy * PADDING_SCALE)
                            cx_max += int(dx * PADDING_SCALE)
                            cy_max += int(dy * PADDING_SCALE)
                        
                        # Classify
                        try:
                            crop_img = original_frame_rgb[cy_min:cy_max, cx_min:cx_max]
                            im_pil = Image.fromarray(crop_img)
                            with torch.no_grad():
                                img = transformer(im_pil).unsqueeze(0)
                                inputs = img.to(device)
                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)
                                label = class_names[preds]
                        except ValueError:
                            continue
                        
                        # Visualize and count buffer
                        if label == 'no_mask':
                            selected_color = COLOR_NO_MASK
                            isNoMask = True
                        elif label == 'incorrect_mask':
                            selected_color = COLOR_INCORRECT
                            isIncorrect = True
                        elif label == 'mask':
                            selected_color = COLOR_MASK
                        else:
                            selected_color = COLOR_INVALID
                        cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), selected_color, 2)

                        if DEBUG:
                            label = 'dx: {} | dy: {}'.format(dx,dy)

                        cv2.putText(image, label, (cx_min, (cy_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=selected_color, thickness=2)

            # End of process and visualization
            end_time = time.time() - start_time

            # Count detected frame
            if isNoMask:
                counter_noMask += 1
            if isIncorrect:
                counter_incorrect += 1

            # Put FPS in frame, optional
            if DEBUG:
                FPS_data.append(1/end_time)
                if len(FPS_data) > 10:
                    # remove old data
                    FPS_data.pop(0)
                FPS_label = np.average(FPS_data)
                cv2.putText(image, 'FPS: {:.2f}'.format(FPS_label), (13, (23)), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.7, color=COLOR_INVALID, thickness=2)
                cv2.putText(image, 'FPS: {:.2f}'.format(FPS_label), (10, (20)), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.7, color=(0,100,255), thickness=2)
            else:
                cv2.putText(image, '# People: {}'.format(people_counter), (13, (23)), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.7, color=COLOR_INVALID, thickness=2)
                cv2.putText(image, '# People: {}'.format(people_counter), (10, (20)), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.7, color=(0,100,255), thickness=2)

            if SAVE_VIDEO:
                # Write the frame into the file 'output.mp4'
                out.write(image)

            # Proceed if need to send email
            if SENDING_EMAIL:
                # save no mask and incorrect mask detection into logs
                if time.time() - email_time < EMAIL_INTERVAL:
                    # save image if more than buffer
                    if counter_noMask > DETECTION_BUFFER or counter_incorrect > DETECTION_BUFFER:
                        if isNoMask and isIncorrect:
                            label_image = 'combined_'
                        elif isNoMask:
                            label_image = 'noMask_'
                        elif isIncorrect:
                            label_image = 'incorrectMask_'
                        full_dir = os.path.join(email.active_folder,current_dir)
                        if not os.path.exists(full_dir):
                            os.makedirs(full_dir)
                        cv2.imwrite(os.path.join(full_dir,label_image+str(int(time.time()))+'.jpg'),image)
                        # flush counter after writing image
                        counter_noMask, counter_incorrect = 0, 0
                else:
                    # change the folder name
                    current_dir = str(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
                    # reset email_time
                    email_time = time.time()
                    # Send trigger to send email
                    # we need to run the recorder in a seperate process, otherwise blocking options
                    # would prevent program from running detection
                    isSendingEmail = True
                    proc = Process(target=sendingEmail)
                    procs.append(proc)
                    proc.start()
                    if DEBUG:
                        print('Create new dir in logs . . .')
            # print('isSendingEmail: {}\tisAlive: {}'.format(isSendingEmail, email_thread.isAlive()))
            cv2.imshow('Frame', image)
            if cv2.waitKey(5) == ord('q'):
                break
    
    if SENDING_EMAIL:
        # terminate all process
        for proc in procs:
            proc.join()
    cap.release()
    cv2.destroyAllWindows()

    if SAVE_VIDEO:
        out.release()
