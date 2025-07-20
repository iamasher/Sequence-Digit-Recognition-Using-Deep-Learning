# **Sequence Digit Recognition using Deep Learning**

This repository contains our **Final Year Major Project**: **Sequence Digit Recognition using Deep Learning**.  
In this project, we extended digit classification capabilities from individual digits to sequences of digits drawn or uploaded as images. The system uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset and includes a **multi-digit detection pipeline** using image processing techniques with **OpenCV**.

This project represents the culmination of our learning journey as students of **Loknayak Jai Prakash Institute of Technology, Chapra**, guided by **Professor Sudhir Pandey**.

---

## **Project Team**

We are final-year B.Tech (CSE) students who collaboratively built this system. Responsibilities are listed by registration number.

| Name            | Registration Number | Roles & Responsibilities                                                                 |
|-----------------|---------------------|--------------------------------------------------------------------------------------------|
| **Md Asher**     | 21105117055         | Model Architecture Design, Training Pipeline, Image Processing, Version Control (GitHub)                                          |
| Rupesh Kumar    | 21105117023         | Visualization, GUI Design, Documentation                                                  |
| Md Abdullah        | 21105117011         | Data Preprocessing, Model Development & Training, GUI Development |
| Fariya Rafat    | 21105117057         | Model Evaluation, Testing, Predictions                                                    |

### **Project Guide**  
**Professor Sudhir Pandey**, Loknayak Jai Prakash Institute of Technology, Chapra.

---

## **Project Overview**

This system implements an end-to-end **digit sequence recognition application**, with the following capabilities:

- Handwritten digit classification using CNNs
- Multi-digit detection using OpenCV (for both drawn and uploaded images)
- Real-time GUI-based interface using **Tkinter**
- Image preprocessing: thresholding, contour detection, dilation, and blurring
- Digit extraction and classification using a trained deep learning model

---

## **Key Features**

-  **High accuracy** (up to 99.6%) on MNIST dataset  
-  **Sequence digit detection** using OpenCV & CNN  
-  **Draw or Upload** digits for real-time predictions  
-  Clean and modular codebase  
-  Training metrics and visualization included  

---

## **Tech Stack**

- **Language**: Python  
- **Framework**: TensorFlow + Keras  
- **Libraries**:  
  - NumPy  
  - OpenCV  
  - Pillow (PIL)  
  - Matplotlib  
  - Seaborn  
- **GUI**: Tkinter  
- **Dataset**: MNIST Handwritten Digits (0–9)

---

## **Requirements**

Save this as `requirements.txt`:

```
tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==9.5.0
matplotlib==3.7.1
seaborn==0.12.2
```

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Setup & Usage**

###  Step 1: Clone the Repo

```bash
git clone https://github.com/iamasher/Sequence-Digit-Recognition-Using-Deep-Learning.git
cd Sequence-Digit-Recognition-Using-Deep-Learning
```

###  Step 2: Train the Model

```bash
python train.py
```

- Trains CNN on MNIST dataset  
- Saves `Trained_Model.h5`  
- Accuracy and loss graphs + confusion matrix saved in `/training_results/`

###  Step 3: Run GUI for Drawing or Upload

```bash
python app.py
```

- Draw digits on canvas or upload an image  
- The model detects, extracts, and predicts all digits in sequence  

---

## **Results & Evaluation**

- **Training Accuracy**: 99.6%  
- **Validation Accuracy**: 99.4%  
- **Loss**: Very low after 10 epochs  
- Supports prediction from real-world noisy and imperfect digit inputs (e.g. drawn or uploaded PNGs)

### Evaluation Metrics Saved:
- Training and validation accuracy/loss plots  
- Confusion matrix  
- Model summary  
- Inference output on uploaded and drawn images

---

## **Contributing**

This is an academic project; contributions are currently limited to team members.  
Feel free to fork, star ⭐ the repo, or raise issues for suggestions.

---

## **Acknowledgments**

We sincerely thank **Professor Sudhir Pandey** for his guidance and support throughout this project.  
Also, we are grateful to **Loknayak Jai Prakash Institute of Technology, Chapra** for the opportunity to work on this problem.

---

## **License**

This repository is for academic purposes only and not intended for commercial use.
