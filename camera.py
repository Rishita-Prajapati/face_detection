import cv2
import numpy as np

# Load cat ears image (ensure it's a transparent PNG)
cat_ears = cv2.imread('cat_ears.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Unable to access the camera")
else:
    print("Press 'q' to close the camera window")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to read frame")
            break

        # Detect face (Haar Cascade or DNN-based face detector)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Resize cat ears to match face width
            resized_ears = cv2.resize(cat_ears, (w, int(w * cat_ears.shape[0] / cat_ears.shape[1])))

            # Get dimensions for overlay
            eh, ew, _ = resized_ears.shape
            roi = frame[y - eh:y, x:x + ew]

            # Blend overlay onto face
            for i in range(eh):
                for j in range(ew):
                    if y - eh + i >= 0 and x + j < frame.shape[1]:
                        alpha = resized_ears[i, j, 3] / 255.0  # Transparency channel
                        for c in range(3):  # Blend RGB channels
                            roi[i, j, c] = (1 - alpha) * roi[i, j, c] + alpha * resized_ears[i, j, c]

        # Show the frame with cat filter
        cv2.imshow('Cat Filter', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close window
    camera.release()
    cv2.destroyAllWindows()
