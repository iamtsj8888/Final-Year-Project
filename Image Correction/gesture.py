from ultralytics import YOLO
import cv2


model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera Off")

while True :
    ret, frame = cap.read()

    if ret == False :
        break

    results = model(frame)[0]

    annotated_img = results.plot(boxes = False, labels = False)
    cv2.imshow("Gestures", annotated_img)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()