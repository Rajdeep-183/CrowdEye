'''from ultralytics import YOLO

# Load the YOLO v8 model
model = YOLO("yolov8n.pt")  # This will automatically download the model if not found

print("YOLOv8 model loaded successfully!")'''

'''import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8l.pt")  # Large version (even better)'''



'''# Load an image
image_path = "D:\Mini Project\Crowd1.jpg.webp"  # Replace with your image file
image = cv2.imread(image_path)

# Run YOLOv8 on the image
results = model(image)

# Draw bounding boxes
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID
        
        if cls == 0:  # Check if detected object is a person
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the result
cv2.imshow("YOLOv8 Human Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''




import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano model (fastest)

# Open a video file or webcam (use 0 for webcam)
video_path = "test.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Define the crowd threshold for stampede prediction
CROWD_THRESHOLD = 16  # Adjust based on experiments

def predict_stampede(people_count):
    """Predicts if a stampede risk is high based on the number of detected people."""
    return people_count > CROWD_THRESHOLD  # If more people than threshold, high risk

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Run YOLOv8 on the frame
    frame = cv2.resize(frame, (640, 640))  # Resize to 640x640 for better detection

    results = model(frame, conf=0.25)  # Lower confidence threshold (default is 0.5)


    # Count number of people detected
    people_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) == 0)

    # Predict stampede risk
    is_stampede_risk = predict_stampede(people_count)
    
    # Display risk level on video
    risk_text = "Stampede Risk: HIGH" if is_stampede_risk else "Stampede Risk: LOW"
    risk_color = (0, 0, 255) if is_stampede_risk else (0, 255, 0)  # Red for HIGH, Green for LOW

    # Draw the risk level and people count on the frame
    cv2.putText(frame, f"People Count: {people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(frame, risk_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, risk_color, 3)

    # Show the frame
    cv2.imshow("Stampede Detection", frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
