# Real-Time Vehicle & Object Detection with YOLOv8

This project implements a real-time object detection, tracking, and counting system based on YOLOv8 and DeepSORT (ReID).
It is designed to detect and track multiple types of objects across video streams from fixed monitors (CCTV) or drones.

🎯 Supported Object Categories

Vehicles: cars, trucks, buses

Pedestrians

Maritime vessels: cruise ships, fishing boats, etc.

🧠 Core Features

Real-time detection using fine-tuned and retrained YOLOv8

Multi-object tracking with DeepSORT + ReID

Object counting across regions or lines (e.g. traffic flow, pedestrian counting)

Persistent ID assignment for each object

Supports live camera, drone footage, and video files

⚙️ Technical Stack

YOLOv8 (custom fine-tuning & retraining)

DeepSORT with Re-identification (ReID)

Python, OpenCV, PyTorch

📊 Use Cases

Traffic flow analysis and vehicle counting

Pedestrian monitoring

Port and maritime surveillance

Smart city and intelligent transportation systems (ITS)

Drone-based aerial monitoring

📌 Output

Real-time bounding boxes and tracking IDs

Per-class counting statistics (cars, people, ships, etc.)

Optional logging and visualization overlays
