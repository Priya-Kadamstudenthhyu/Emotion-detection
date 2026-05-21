# emotion_realtime.py
import cv2
from deepface import DeepFace

# ----------------------------
# Initialize webcam
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 is default webcam
# Uncomment below if 0 doesn't work
# cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# ----------------------------
# Real-time emotion detection
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (480, 360))

    try:
        # Analyze emotion (enforce_detection=False avoids errors if face not detected)
        result = DeepFace.analyze(resized_frame, actions=["emotion"], enforce_detection=False)

        # DeepFace returns a list sometimes
        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "N/A")

        # Draw the dominant emotion on the frame
        cv2.putText(frame,
                    f"{dominant_emotion}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

    except Exception as e:
        # In case no face is detected
        cv2.putText(frame,
                    "No face detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
