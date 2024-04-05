from ultralytics import YOLO
import cv2
import numpy as np


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
    for r in results:
        if(int(r.boxes.cls.item()) == 0):
            text = f"{text} \n----------- {r.boxes.cls.item()} -----------\n"
            text = text + str(r.masks) + "\n"
            masks_points = r.masks.xy
            annotated_img = r.plot(boxes = False, labels = False)
            break

    # with open("masks.txt", "w")as file:
    #     file.write(text)
        
    # annotated_img = frame
    # if(len(masks_points) != 0):
    #     masks_points.append(masks_points[0])
    #     masks_points = np.array(masks_points, np.int32)
    #     masks_points = masks_points.reshape((-1, 1, 2))
    #     print(type(masks_points))
    #     annotated_img = cv2.fillPoly(frame,masks_points,color=(0,0,255))

    
    cv2.imshow("Segmentation", annotated_img)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break


