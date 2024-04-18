from ultralytics import YOLO
import cv2
import numpy as np
import torch


# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model


def masks_list():
   print("TO_DO")

def process(results, frame):
    device = torch.device('cuda:0')

    input_tensor = torch.from_numpy(frame)
    input_tensor = input_tensor.to(device)

    mask_img = np.zeros(shape=(384, 640))
    mask_tensor = torch.from_numpy(mask_img)
    mask_tensor = mask_tensor.to(device=device)

    annotated_img = results.plot(boxes=False, labels = False)

    foreground_img =  np.zeros(shape=(384, 640, 3), dtype=np.uint8)
    foreground_img = foreground_img.reshape((384,640,3))
    foreground_tensor = torch.from_numpy(foreground_img)
    foreground_tensor = foreground_tensor.to(device)
    
    for r in results:
        mask_tensor = torch.logical_or(torch.eq(r.masks.data[0], 1), torch.eq(mask_tensor,1)).to(torch.float32)
    
    mask_img = mask_tensor.cpu().numpy()

    # with open("foreground.txt" , "w") as file :
    #     file.write(str(foreground_tensor))
    #     file.write(f"\n\n ---- {foreground_tensor.shape} ---- \n\n")

    # with open("input_tensor.txt", "w") as file :
    #     file.write(str(input_tensor))
    #     file.write(f"\n\n ---- {input_tensor.shape} ---- \n\n")

    foreground_tensor[:,:,0] = torch.mul(torch.logical_not(torch.eq(mask_tensor,0)).to(torch.float32) , input_tensor[:,:,0])
    foreground_tensor[:,:,1] = torch.mul(torch.logical_not(torch.eq(mask_tensor,0)).to(torch.float32) , input_tensor[:,:,1])
    foreground_tensor[:,:,2] = torch.mul(torch.logical_not(torch.eq(mask_tensor,0)).to(torch.float32) , input_tensor[:,:,2])

    foreground_img = foreground_tensor.cpu().numpy()

    return mask_img, annotated_img, foreground_img


if __name__ == "__main__":
        
    path = "C:/Users/TEJAS BIBEKAR/Downloads/2.mp4"
    
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    
    size = (frame_width, frame_height) 

    while True :
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 384))

        if not ret :
            break
        
        results = model(frame)[0]

        mask_img, annotated_img, foreground_img = process(results, frame)

        cv2.imshow("Mask", mask_img)  
        cv2.imshow("Foreground" , foreground_img)
        cv2.imshow("Segmentation", annotated_img)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) == ord('q'):
            break


    cap.release() 
    cv2.destroyAllWindows() 