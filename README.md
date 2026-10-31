
# "Baka Naman May Bakante"
### A Smart Study Space Occupancy Analytics and Recommendation System Using YOLO-Based Computer Vision and Predictive Modeling

**Baka Naman May Bakante** is a computer vision-based study space occupancy analytics and recommendation system designed to help students identify suitable study spaces based on predicted occupancy, room capacity, and individual preferences.

The project integrates **YOLO-based people detection, time-series forecasting, occupancy analytics, and a congestion-aware recommendation engine** into a web-based dashboard. It aims to provide students with a more informed way to select where and when to study while helping distribute demand across available spaces.

Developed as an undergraduate capstone project at **De La Salle University, Manila**, this study adopts a software-only, offline proof-of-concept approach using publicly available datasets to evaluate the feasibility of the proposed system before potential real-world deployment.

---

## Project Overview

The system consists of four interconnected components:

1. **Occupancy Detection** – Uses YOLO object detection models to identify and count people in recorded classroom and library images and videos.
2. **Occupancy Analytics** – Processes historical occupancy records to identify utilization patterns, peak hours, and room availability.
3. **Occupancy Forecasting** – Uses machine learning and time-series models to predict future occupancy at 15-, 30-, and 60-minute intervals.
4. **Study Space Recommendation** – Recommends suitable study spaces based on predicted occupancy, room capacity, user preferences, and concurrent user demand.

The resulting information is presented through an interactive web dashboard featuring a virtual occupancy map, historical analytics, occupancy forecasts, and personalized recommendations.

## Key Features

### 1. YOLO-Based People Detection
- Benchmarks YOLOv8, YOLO11, and YOLO26 nano and small models.
- Evaluates pretrained and fine-tuned models using publicly available datasets.
- Compares inference performance using PyTorch, TensorRT FP16, and CPU configurations.
- Uses YOLO26n at 1280-pixel input resolution as the selected occupancy detection model.

### 2. Occupancy Analytics
- Monitors occupancy across six virtual study spaces.
- Generates 15-minute occupancy records from historical datasets.
- Analyzes average occupancy, peak occupancy, and room utilization.
- Identifies frequently occupied periods and underutilized spaces.
- Presents historical occupancy patterns through interactive visualizations.

### 3. Predictive Occupancy Modeling
- Forecasts occupancy 15, 30, and 60 minutes ahead.
- Implements LightGBM and Prophet forecasting models.
- Evaluates predictions against persistence and seasonal-naive baselines.
- Incorporates predicted occupancy and room capacity to estimate future availability.

### 4. Smart Study Space Recommendation
- Filters study spaces according to intended activity, quietness, and outlet preferences.
- Ranks suitable rooms based on predicted occupancy and available capacity.
- Returns the top three recommended study spaces with explanations.
- Implements shared soft reservations to account for concurrent users and reduce overcrowding.
- Considers both expected and peak predicted occupancy when ranking recommendations.

### 5. Interactive Web Dashboard
- Student interface with a virtual occupancy map and room availability cards.
- Occupancy forecast visualizations and recommendation forms.
- Administrator dashboard for historical utilization analytics and forecast evaluation.
- Camera demonstration page for replaying YOLO-based detections.
- Simulated clock for navigating historical occupancy records.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Computer Vision | Ultralytics YOLO, OpenCV |
| Deep Learning | PyTorch, CUDA |
| Model Optimization | NVIDIA TensorRT FP16 |
| Machine Learning | LightGBM, Prophet |
| Data Processing | Pandas, NumPy |
| Database | MySQL 8.0 |
| Database Integration | SQLAlchemy |
| Web Application | Streamlit |
| Data Visualization | Plotly |
| Containerization | Docker, Docker Compose |
| Cloud Deployment | Microsoft Azure Virtual Machine |
| Testing | Pytest |

---

## Datasets

This project uses publicly available datasets for person detection, people counting, and occupancy analysis. The datasets were used for academic and research purposes, subject to their respective licenses and citation requirements.

| Dataset | Source |
|---|---|
| CEPDOF (Challenging Events for Person Detection from Overhead Fisheye Images) | [Boston University - CEPDOF](https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/) |
| FRIDA (Fisheye Re-Identification Dataset with Annotations) | [Boston University - FRIDA](https://vip.bu.edu/projects/vsns/cossy/datasets/frida/) |
| CO2FISHEYE | [Boston University - CO2FISHEYE](https://vip.bu.edu/projects/vsns/cossy/datasets/co2fisheye/) |
| SCUT-HEAD | [GitHub - SCUT-HEAD Dataset Release](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) |
| RHC (Room Human Counting) | [Mendeley Data - RHC](https://data.mendeley.com/datasets/vt5c8h6kmh/1) |
| AIRC-SMARTCLASS | [Mendeley Data - Part 3](https://data.mendeley.com/datasets/kk93h9zzbc/1) |
| ROBOD (Room-level Occupancy and Building Operation Dataset) | [GitHub - ROBOD](https://github.com/ideas-lab-nus/robod) |
| SDU Occupancy Presence and Trajectory Dataset | [University of Southern Denmark](https://portal.findresearcher.sdu.dk/en/datasets/dataset-occupancy-presence-and-trajectory-data-from-an-instrument/) |
| Library Seats | [Roboflow - Library Seats](https://universe.roboflow.com/natnaels-workspace-5sk2c/library-seats)

**Dataset Usage and Attribution**

- All datasets remain the property of their respective creators and institutions.
- Users must review and comply with the original dataset licenses and terms of use.
- Appropriate citations and acknowledgments must be provided when using these datasets.
- The datasets are not redistributed as part of this repository unless explicitly permitted by their respective licenses.

---

## Key Results

| Evaluation | Result |
|---|---|
| Selected Detection Model | YOLO26n at 1280 pixels |
| Macro Counting MAE | 9.72 people/frame |
| TensorRT FP16 Inference Time | 8.56 ms/frame |
| Forecasting Improvement | 29%–58% lower 15-minute MAE than persistence in four of five evaluable spaces |
| Recommendation Success Rate | 72.4% across 377 test requests |
| Random Recommendation Baseline | 56.0% |
| Static Recommendation Baseline | 52.5% |
| Functional Testing | 25/25 automated tests passed |

### Recommendation and Congestion Management

The recommendation engine successfully selected the least-occupied suitable room in 72.4% of evaluated requests.

Shared soft reservations also reduced the proportion of students assigned to full rooms during simulated concurrent demand:

- **50 simultaneous requests:** Reduced overflow from 4.1% to 0.2%.
- **100 simultaneous requests:** Reduced overflow from 25.9% to 6.6%.

These results demonstrate the potential of integrating occupancy forecasting and shared recommendations to improve study-space allocation.

---

## System Scope and Limitations

This project was developed and evaluated as a **software-only, offline proof of concept**.

- Uses publicly available recorded datasets rather than live campus camera feeds.
- Occupancy records represent six virtual study spaces derived from existing datasets.
- Computer vision detection and forecasting are executed offline.
- The deployed dashboard presents historical data through a simulated clock.
- The system does not perform live occupancy monitoring or actual room reservations.
- Real-world campus deployment, user studies, and live camera integration remain areas for future development.

---

## Future Improvements

Potential extensions of the project include:

- Integration of real-time camera feeds and on-device people counting.
- Calibration of detection models using actual campus study-space environments.
- Expansion to additional university facilities and study spaces.
- Improved forecasting using longer academic-term occupancy records.
- Implementation of secure user authentication and actual reservation functionality.
- Further usability, accessibility, reliability, and security evaluation.

---

## Author

**Angela Mari D. Romero**

BS Manufacturing Engineering and Management  
Specialization in Mechatronics and Robotics  
De La Salle University – Manila

**Project Type:** Undergraduate Capstone Project  
**Research Areas:** Computer Vision | Artificial Intelligence | Machine Learning | Predictive Analytics | Process Automation | Smart Campus Systems
