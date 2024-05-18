import cv2
from mtcnn import MTCNN

# Initialize the front-facing camera
cap = cv2.VideoCapture(1)  # 0 is usually the default camera, 1 is typically the front-facing camera on macOS

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Initialize MTCNN model
mtcnn = MTCNN()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Detect faces in the frame
    faces = mtcnn.detect_faces(frame)

    # Draw bounding boxes around the detected faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
