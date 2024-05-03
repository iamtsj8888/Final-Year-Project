# Major Project
# Deep Learning Based Face Extraction and Enhancement for Video Survellience 

# Project Mentor -- Jitendra Bharadwaj (Assistant Prof.)
# Completed by -
#   Tejas Bibekar 20106010
#   Tanmay Giram 20106021
#   Nishant Wankhade 20106070 

# Github link --> https://github.com/iamtsj8888/Final-Year-Project

from ultralytics import YOLO
import cv2
import numpy as np
import math
import torch
import os

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
    """ 
        Loads the model and runs inference on the given frame img
        Args:
            image : Frame read from the video capture
    """

    # Load model from a path 
    model = YOLO('Project/Code/Face extraction/Model/yolov8n-pose.pt')  

    # Predict with the model
    results = model(img)

    top = []
    for r in results:
        # Only Keypoints of Eyes, Ears, Nose
        top = np.array(r.keypoints.xy[0][0:5].cpu().numpy())
        break
    
    arr = []
    print(len(top))
    for i in top:
        arr.append([int(i[0]), int(i[1])])
    
    img = results[0].plot(boxes = False, labels = False, kpt_line = False)
    return [img , arr]


def distance_between(p1, p2):
    """ 
        Calculates the Euclidean distance between those 2 points
        Args:
            2 points of class Point
    """
    return int(math.sqrt((p1[X_coord]-p2[X_coord]) * (p1[X_coord]-p2[X_coord]) + (p1[Y_coord]-p2[Y_coord]) * (p1[Y_coord]-p2[Y_coord])))


def extract_face(img, keypoints):
    """ 
        Calculates the Euclidean distance between those 2 points
        Args:
            Input image frame,
            Keypoints detected by the Pose detection model
    """

    if(keypoints.nose == [0,0]) :
        face_img = img[0:0, 0:0]
        return face_img
    

    if(keypoints.eye["l"] == [0,0] and keypoints.eye["r"] == [0,0]):
        face_img = img[0:0, 0:0]
        return face_img

    nose_y = keypoints.nose[Y_coord]
    nose_x = keypoints.nose[X_coord]
    
    eye_l_x = keypoints.eye["l"][X_coord]
    eye_l_y = keypoints.eye["l"][Y_coord]
    eye_r_x = keypoints.eye["r"][X_coord]
    eye_r_y = keypoints.eye["r"][Y_coord]

    if(keypoints.eye["l"] != [0,0]):
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

    elif(keypoints.eye["r"] != [0,0]):
        width = int(2 * (distance_between(keypoints.nose, keypoints.eye["r"])))
        height = int(width * 1.5)

        start_x = nose_x - width
        start_y = eye_r_y - height

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

def save_face_img(face_img, frame_img, annotated_img, nth_dir, count):
    """ 
        Saves the face image, original image, annotated image, count of the current image
        Args:
            face image, original image, annotated image, count of the current image
    """


    os.makedirs(f"Result{nth_dir}/face", exist_ok=True)
    os.makedirs(f"Result{nth_dir}/frame", exist_ok=True)
    os.makedirs(f"Result{nth_dir}/annotated_frame", exist_ok=True)
    cv2.imwrite(f"Result{nth_dir}/face/{count}.png", face_img)
    cv2.imwrite(f"Result{nth_dir}/frame/{count}.png", frame_img)
    cv2.imwrite(f"Result{nth_dir}/annotated_frame/{count}.png", annotated_img)

def result_dir():
    """
        Determines the result output folder 
    """
    nth_dir = 1
    while(os.path.exists(f"Result{nth_dir}")):
        nth_dir = nth_dir + 1
    
    return nth_dir

def process(path):
    """ 
        Starts the video using OpenCv VideoCapture object
        Args:
            path to the video
    """

    cap = cv2.VideoCapture(path)

    nth_dir = result_dir()

    if not cap.isOpened():
        print("Camera Off")

    face_list = []
    global key_img

    count = 1
    while True :
        ret, frame = cap.read()

        if ret == False :
            break

        annotated_img, keypoints = extract_points(frame)
        
        if(len(keypoints) < 5): 
            continue

        points = Points(keypoints=keypoints)

        face_image = extract_face(frame, points)
        
        key_img = face_image

        cv2.imshow("Annotated Image", annotated_img)
        
        if(face_image.shape != (0,0,3)):
            print(f"---------------------- {count} ------------------------")
            points.show_points()
            cv2.imshow("Face Extraction", face_image)
            save_face_img(frame_img=frame,annotated_img=annotated_img, face_img=face_image, count= count, nth_dir=nth_dir)
            count = count + 1

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    # Path to the Video or 0 for webcam
    path = "../1.mp4"

    # process(path=0)
    