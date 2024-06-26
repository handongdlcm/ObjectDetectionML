import time
import cv2

# Initialize the front-facing camera
cap = cv2.VideoCapture(1)  # 0 is usually the default camera, 1 is typically the front-facing camera on macOS

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()


# Variables for benchmarking
prev_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Calculate framerate
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
    prev_time = current_time

    # Add framerate to the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
