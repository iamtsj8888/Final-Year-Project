import cv2
import numpy as np


# Create the KNN background subtractor
knn = cv2.createBackgroundSubtractorKNN()

# Open the video file
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Loop over the frames of the video
while True:
    # Read the frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    frame = cv2.resize(frame, (800,800))

    # Apply the KNN background subtractor to the frame
    fgmask = knn.apply(frame)

    kernal = np.ones((5,5), np.uint8)

    fgmask = cv2.erode(fgmask, kernal, iterations = 2)
    fgmask = cv2.dilate(fgmask, kernal, iterations=2)

    cv2.rectangle(frame, (10,2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    fgmask[np.abs(fgmask) < 250] = 0

    # Show the foreground mask
    cv2.imshow('Frame', frame)
    cv2.imshow('FG mask', fgmask)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video file and destroy the windows
cap.release()
cv2.destroyAllWindows()