from ultralytics import YOLO
import cv2
from PIL import Image


# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model

cap = cv2.VideoCapture(0)


while True :
    ret, frame = cap.read()

    if not ret :
        break
    
    results = model(frame)[0]
    for r in results:
        print(r.masks)
        break

    annotated_img = results[0].plot()
    cv2.imshow("Segmentation", annotated_img)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break


