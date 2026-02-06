🌾 AgroPred - agropred.netlify.app
Enhancing Wheat Breeding with Multi-Trait Genomic Prediction & AI-Based Crop Monitoring.

AgroPred is an AI-driven agricultural intelligence platform designed to assist researchers, breeders, and agronomists in predicting wheat phenotypic traits using genomic data and monitoring wheat development stages using images.
The project integrates machine learning, deep learning, image processing, and a web-based system to provide actionable insights for modern wheat breeding programs.

🚀 Project Overview

AgroPred consists of two independent but complementary AI modules:

1️⃣ Genomic Prediction Module

Predicts multiple wheat phenotypic traits directly from DNA / SNP marker sequences using machine learning models.

2️⃣ Wheat Development Stage Monitoring Module

Analyzes wheat images to detect spikes and classify the crop growth stage using deep learning.

🧩 System Architecture: 

User Input
   ├── DNA / SNP Sequence
   │       ↓
   │   ML Models (XGBoost)
   │       ↓
   │   Phenotypic Trait Predictions
   │
   └── Wheat Image
           ↓
     Spike Detection (Faster R-CNN)
           ↓
   Growth Stage Classification (CNN)
🧬 Module 1: Genomic Prediction
🔹 Input

SNP/DNA sequence encoded as numerical markers

Fixed-length input (e.g., 24-character SNP representation)

🔹 Output

Predicted wheat phenotypic traits:
Grain Filling Duration (GFD)
Grain Number per Spike (GNPS)
Grain Weight per Spike (GWPS)
Plant Height (PH)
Grain Yield (GY)
Additional nutritional traits (Fe, Zn, TKW)

🔹 Models Used

XGBoost Regressor
Separate trained model for each trait

🔹 Dataset

Multi-location wheat phenotypic data

Locations include:
  Karnal
  Ludhiana
  IARI Delhi
  IARI Jharkhand
  Dharwad
🌱 Module 2: Wheat Development Stage Monitoring
🔹 Input

Wheat field or spike images

🔹 Processing Pipeline

Wheat Spike Detection

Model: ResNet50 + Faster R-CNN

Output: Bounding boxes around wheat spikes

Growth Stage Classification

Model: CNN

Classes:
  Filling
  Filling–Ripening
  Post-Filling

🔹 Output

Detected wheat spikes

Classified growth stage of wheat

Technologies Used
🔹 Machine Learning & Deep Learning
  Python
  XGBoost
  PyTorch
  TensorFlow
  CNN
  Faster R-CNN (ResNet50 backbone)
🔹 Backend
  Flask
  REST APIs
  JWT-based authentication
  Email verification
  Model serialization (.pkl files)
🔹 Database
  MongoDB
🔹 Frontend
  React

🔐 Key Features

Multi-trait genomic prediction
Image-based wheat spike detection
Crop growth stage monitoring
Secure user authentication
Email notifications
Scalable backend architecture
Research-oriented, modular design

🧪 Use Cases

Wheat breeding programs
Crop research institutes
Agronomists and agricultural scientists
Decision support for yield improvement
Early-stage crop development analysis

📊 Results & Performance

Accurate prediction of multiple phenotypic traits
Robust spike detection across multiple datasets
Reliable classification of wheat development stages
Handles data from diverse geographic locations
