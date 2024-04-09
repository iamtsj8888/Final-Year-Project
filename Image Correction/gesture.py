from ultralytics import YOLO
import cv2
import numpy as np

X_coord = 0
Y_coord = 0

class Points:
    def __init__(self, keypoints) -> None:
        if(len(keypoints) != 7) : print("Missing some KeyPoint")
        # Keypoints given by YOLOv8 Pose model
        # Nose --> 0 index
        # Left Eye --> 1 , Right Eye --> 2
        # Left Ear --> 3, Right Ear --> 4
        # Left Shoulder --> 5, Right shoulder --> 6
        self.nose = keypoints[0]
        self.eye = { "l" : keypoints[1], "r" : keypoints[2] }
        self.ear = { "l" : keypoints[3], "r" : keypoints[4] }
        self.shoulder = { "l" : keypoints[5], "r" : keypoints[6] }

    def show_points(self):
        print(f"""
              Nose --> {self.nose},\n
              Eyes :-> Left {self.eye["l"]}, Right {self.eye["r"]}\n
              Ears :-> Left {self.ear["l"]}, {self.ear["r"]}\n
              Shoulders :-> Left {self.shoulder["l"]}, Right {self.shoulder["r"]}""")


def extract_points(img):
    
    # Load model
    model = YOLO('Model/yolov8n-pose.pt')  

    # Predict with the model
    results = model(img)

    top = []
    for r in results:
        # Only Keypoints of Eyes, Ears, Nose, Shoulders 
        top = np.array(r.keypoints.xy[0][0:7].cpu().numpy())
        break
    
    arr = []
    print(len(top))
    for i in top:
        arr.append([int(i[0]), int(i[1])])
        if(i[0] != 0 and i[1] != 0) :
            cv2.circle(img, (int(i[0]), int(i[1])), radius=5, color=(0, 255, 0), thickness=-1)

    return [img , arr]


def extract_face(img, keypoints):
    start_x = keypoints.shoulder["r"][X_coord]
    end_x = keypoints.shoulder["l"][X_coord]

    end_y = keypoints.shoulder["l"][Y_coord]
    nose_y = keypoints.nose[Y_coord]

    start_y = min(0,end_y - 2 * (end_y - nose_y))

    face_img = img[start_y:end_y, start_x:end_x]

    return face_img


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera Off")

    while True :
        ret, frame = cap.read()

        if ret == False :
            break

        annotated_img, keypoints = extract_points(frame)

        points = Points(keypoints=keypoints)
        points.show_points()

        face_image = extract_face(frame, points)

        cv2.imshow("Annotated Image", annotated_img)
        cv2.imshow("Face Extraction", face_image)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()