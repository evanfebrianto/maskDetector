from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import mediapipe as mp
from lib import helper

parser = helper.parse_config('config.ini')

SHORT_RANGE = parser['BOOLEAN']['SHORT_RANGE']
DEBUG = parser['BOOLEAN']['DEBUG']
SAVE_VIDEO = parser['BOOLEAN']['SAVE_VIDEO']

WIDTH = parser['INT']['WIDTH']
HEIGHT = parser['INT']['HEIGHT']

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
print('SAVE_VIDEO: {}\n'.format(SAVE_VIDEO))
print('WIDTH: {}'.format(WIDTH))
print('HEIGHT: {}\n'.format(HEIGHT))
print('PADDING_SCALE: {}'.format(PADDING_SCALE))
print('DETECTION_CONFIDENCE: {}\n'.format(DETECTION_CONFIDENCE))
print('MEAN: {}'.format(MEAN))
print('STD: {}\n'.format(STD))
print('COLOR_MASK: {}'.format(COLOR_MASK))
print('COLOR_INCORRECT: {}'.format(COLOR_INCORRECT))
print('COLOR_NO_MASK: {}'.format(COLOR_NO_MASK))
print('COLOR_INVALID: {}'.format(COLOR_INVALID))
print('{}\n'.format('*'*50))


# For webcam input:
cap = cv2.VideoCapture(2)
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
        original_frame_rgb = image.copy()
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)


        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                cx_min, cy_min, cx_max, cy_max = helper.get_bb(image, detection)
                if cx_min is not None:
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
                    
                    # Visualize
                    if label == 'no_mask':
                        selected_color = COLOR_NO_MASK
                    elif label == 'incorrect_mask':
                        selected_color = COLOR_INCORRECT
                    elif label == 'mask':
                        selected_color = COLOR_MASK
                    else:
                        selected_color = COLOR_INVALID
                    cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), selected_color, 2)

                    if DEBUG:
                        dimension = 'dx: {} | dy: {}'.format(dx,dy)
                        cv2.putText(image, dimension, (cx_min, (cy_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=selected_color, thickness=2)
                    else:
                        cv2.putText(image, label, (cx_min, (cy_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=selected_color, thickness=2)
                
                if SAVE_VIDEO:
                    # Write the frame into the file 'output.mp4'
                    out.write(image)

        cv2.imshow('Frame', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows() 
