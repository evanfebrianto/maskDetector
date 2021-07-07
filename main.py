from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
PADDING = 25
DEBUG = False

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(2)

transformer = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    original_frame_rgb = image.copy()
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            if DEBUG:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

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
            cx_min -= PADDING
            cy_min -= PADDING
            cx_max += PADDING
            cy_max += PADDING
            
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
                cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (0, 0, 255), 2)
            elif label == 'incorrect_mask':
                cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (0, 240, 255), 2)
            else: # label == mask
                cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 0), 2)
            if DEBUG:
                cv2.imshow('Faces-'+str(i), crop_img)
    cv2.imshow('Frame', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows() 
