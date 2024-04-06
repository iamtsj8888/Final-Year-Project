from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model

cap = cv2.VideoCapture(0)


while True :
    ret, frame = cap.read()

    if not ret :
        break
    
    results = model(frame)[0]
    text = ""
    masks_points = []

    annotated_img = frame
    mask_img = frame
    for r in results:
        if(int(r.boxes.cls.item()) == 0):
            text = f"{text} \n----------- {r.boxes.cls.item()} -----------\n"
            text = text + str(r.masks) + "\n"
            masks_points = r.masks.xy
            annotated_img = r.plot(boxes = False, labels = False)
            
            mask_img = r.masks.data[0].cpu().numpy()
            break
    
    cv2.imshow("Mask", mask_img)    
    cv2.imshow("Segmentation", annotated_img)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break


