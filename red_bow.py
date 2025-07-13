import cv2

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera (0 is usually the default camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access the camera")
else:
    print("Press 'q' to close the camera window")

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        if not ret:
            print("Error: Unable to read frame")
            break

        frame = cv2.flip(frame, 1)
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        


        # Draw red rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Red color (BGR: (0, 0, 255))

        # Display the live camera feed
        cv2.imshow('Face Detection', frame)

        # Exit the camera feed when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()
