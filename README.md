# CrowdEye

## 🚀 Overview
This project leverages **YOLOv8** for real-time human detection and **stampede prediction** based on crowd density. By analyzing video frames, it detects individuals and determines if the **crowd size exceeds a defined threshold**, thereby predicting a potential stampede risk.

---

## 🔥 Features
✅ Real-time human detection using **YOLOv8**  
✅ Custom threshold to predict **stampede risk**  
✅ Works with both **images** and **videos**  
✅ Supports **live webcam feed** for real-time monitoring  
✅ Easy-to-customize parameters  

---

## 🛠️ Steps to Run the Project

### **Step 1️⃣: Setting Up YOLO Environment**
1. Install Python (Recommended: **Python 3.8+**)
2. Install required dependencies using pip:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

### **Step 2️⃣: Choosing the Right Model**
We chose **YOLOv8n** (**Nano Version**) as it is **fastest** and lightweight, making it ideal for real-time detection on **low-end machines**.

Other available versions: **YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x** (larger models provide better accuracy but are slower).

### **Step 3️⃣: Importing Required Libraries**
In your Python script, import the necessary modules:
   ```python
   import cv2
   from ultralytics import YOLO
   ```

### **Step 4️⃣: Processing Human Detection on an Image**
1. Load the YOLO model:
   ```python
   model = YOLO("yolov8n.pt")
   ```
2. Read the image and run detection:
   ```python
   image = cv2.imread("image.jpg")
   results = model(image)
   ```
3. Draw bounding boxes around detected **humans**.

### **Step 5️⃣: Setting Up a Threshold for Stampede Prediction**
- Define a **crowd threshold** (e.g., 16 people).
- If the detected people count **exceeds this threshold**, the system raises a **stampede risk alert**.
- The output video frame displays:
  - **People count**
  - **Stampede risk level** (LOW/HIGH)

#### **Code Snippet:**
```python
CROWD_THRESHOLD = 16

def predict_stampede(people_count):
    return people_count > CROWD_THRESHOLD  # High risk if count exceeds threshold
```

---

## 📌 Running the Stampede Prediction on Video
To analyze a **video** or **webcam feed**, run:
```python
python stampede_detection.py
```
Press **'Q'** to exit the video window.

---

## 📷 Sample Output
🔹 **People Count: 18**  
🔹 **Stampede Risk: HIGH (⚠️ Red Alert)**  
🔹 Real-time detection bounding boxes drawn around humans

---

## 📩 Future Enhancements
🔹 Integrate an **alert system** (sound or email notifications)  
🔹 Improve accuracy using **YOLOv8m/l models** for better detection  
🔹 Deploy as a **web or mobile application** for real-world monitoring  

---

## 📜 License
This project is open-source. Feel free to use, modify, and enhance it!

---

## 🎯 Contribute
We welcome contributions! Fork the repo and submit your pull requests. Let's build an **AI-powered safety system** together! 💡🚀

