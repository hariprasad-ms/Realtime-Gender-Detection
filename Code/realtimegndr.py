# Import necessary libraries
import cvlib as cv
import cv2
import numpy as np

# Access the webcam
webcam = cv2.VideoCapture(0)

# Set padding for bounding box
padding = 20

#Start the loop to read video from the webcam
while webcam.isOpened():

# Read the current frame
    status, frame = webcam.read()

    # Detect the faces in the frame
    face, confidence = cv.detect_face(frame)

    # Loop over each face and draw a bounding box around it
    for idx, f in enumerate(face):        
        
        # Determine the start and end coordinates for the bounding box
        (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
        (endX,endY) = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
        
        # Draw the bounding box around the face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        
        # Extract the face region
        face_crop = np.copy(frame[startY:endY, startX:endX]) 
        
        # Detect the gender of the face
        (label, confidence) = cv.detect_gender(face_crop)
        
        # Determine the gender label with the highest confidence score
        idx = np.argmax(confidence)
        label = label[idx]
        
        # Format the gender label and confidence score
        label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
        
        # Determine the position to display the gender label
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        
        # Draw the gender label on the frame
        cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Display the frame with bounding boxes and gender labels
    cv2.imshow("Real-time gender detection", frame)

    # Press "s" to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break


# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()