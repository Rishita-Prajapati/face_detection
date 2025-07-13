import cv2
import numpy as np

# Load the pre-trained Haar cascade classifier for face detection.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame.
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over detected faces.
    for (x, y, w, h) in faces:
        # Extract the region of interest (the face region).
        face_roi = frame[y:y + h, x:x + w]

        # Apply a Gaussian blur to the face region.
        face_roi_blur = cv2.GaussianBlur(face_roi, (99, 99), 30)

        # Replace the original face region with the blurred version.
        frame[y:y + h, x:x + w] = face_roi_blur

        # Optionally, draw a rectangle (unblurred) to show the detected face region.
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame.
    cv2.imshow('Real-Time Face Blur', frame)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows.
cap.release()
cv2.destroyAllWindows()
