from ultralytics import YOLO
import cv2
import numpy as np
import math
import torch

X_coord = 0
Y_coord = 1

class Points:
    def __init__(self, keypoints) -> None:
        if(len(keypoints) != 5) : print("Missing some KeyPoint")
        # Keypoints given by YOLOv8 Pose model
        # Nose --> 0 index
        # Left Eye --> 1 , Right Eye --> 2
        # Left Ear --> 3, Right Ear --> 4
        # Left Shoulder --> 5, Right shoulder --> 6
        self.nose = keypoints[0]
        self.eye = { "l" : keypoints[1], "r" : keypoints[2] }
        self.ear = { "l" : keypoints[3], "r" : keypoints[4] }
        # self.shoulder = { "l" : keypoints[5], "r" : keypoints[6] }

    def show_points(self):
        print(f"""
              Nose --> {self.nose},\n
              Eyes :-> Left {self.eye["l"]}, Right {self.eye["r"]}\n
              Ears :-> Left {self.ear["l"]}, Right {self.ear['r']}\n""")


def extract_points(img):
    
    # Load model
    model = YOLO('Model/yolov8n-pose.pt')  

    # Predict with the model
    results = model(img)

    top = []
    for r in results:
        # Only Keypoints of Eyes, Ears, Nose, Shoulders 
        top = np.array(r.keypoints.xy[0][0:5].cpu().numpy())
        break
    
    arr = []
    print(len(top))
    for i in top:
        arr.append([int(i[0]), int(i[1])])
    
    img = results[0].plot(boxes = False, labels = False, kpt_line = False)
    return [img , arr]


def distance_between(p1, p2):
    return int(math.sqrt((p1[X_coord]-p2[X_coord]) * (p1[X_coord]-p2[X_coord]) + (p1[Y_coord]-p2[Y_coord]) * (p1[Y_coord]-p2[Y_coord])))


def extract_face(img, keypoints):
    nose_y = keypoints.nose[Y_coord]
    nose_x = keypoints.nose[X_coord]
    
    eye_l_x = keypoints.eye["l"][X_coord]
    eye_l_y = keypoints.eye["l"][Y_coord]
    eye_r_x = keypoints.eye["r"][X_coord]
    eye_r_y = keypoints.eye["r"][Y_coord]

    width = int(2 * (distance_between(keypoints.nose, keypoints.eye["l"])))
    height = int(width * 1.5)

    start_x = nose_x - width
    start_y = eye_l_y - height

    end_x = nose_x + width
    end_y = nose_y + height

#     print(f"""
# Start X --> {start_x},
# End X --> {end_x},
# Start Y --> {start_y},
# End Y --> {end_y},
# """)
    
    face_img = img[start_y:end_y, start_x:end_x]

    return face_img

def save_face_img(face_img, key_img):
    device = torch.device('cuda:0')
    
    key_img_tensor = torch.from_numpy(key_img)
    key_img_tensor = key_img_tensor.to(device)

    face_img_tensor = torch.from_numpy(face_img)
    face_img_tensor = face_img_tensor.to(device)

    diff = torch.subtract(face_img_tensor, key_img_tensor)

    print(diff)



if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera Off")

    face_list = []
    global key_img

    while True :
        ret, frame = cap.read()

        if ret == False :
            break

        annotated_img, keypoints = extract_points(frame)

        points = Points(keypoints=keypoints)
        # points.show_points()

        face_image = extract_face(frame, points)
        
        if(len(face_list) == 0):
            face_list.append(face_image)
            key_img = face_image
        else:
            save_face_img(key_img=key_img, face_img=face_image)

        cv2.imshow("Annotated Image", annotated_img)
        cv2.imshow("Face Extraction", face_image)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()