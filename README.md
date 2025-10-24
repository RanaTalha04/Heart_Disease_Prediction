# 🩺 Heart Disease Prediction using Machine Learning

This project aims to predict the likelihood of heart disease based on key health indicators such as age, cholesterol level, blood pressure, and other medical attributes.  
The goal is to assist in early diagnosis and prevention by applying machine learning techniques to analyze patient data.

---

## 📊 Overview

Heart disease is a leading global health issue. Early prediction through data-driven models can assist in medical diagnosis and preventive care.  
This project uses supervised learning algorithms to analyze patient data and predict the risk of heart disease.

---

## 🌐 Live Demo
🚀 **Try it here:** [Heart Disease Prediction App](https://heart-disease-prediction-0.streamlit.app/) 

---

## 🧠 Machine Learning Models Used
- Logistic Regression  
- Decision Tree
- Naive Bayes
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

Each model was trained and compared using metrics such as accuracy and F1-score.

---

## ⚙️ Features

- Data preprocessing (handling missing values, normalization, encoding)
- Exploratory Data Analysis (EDA) with visualizations
- Feature selection and correlation analysis
- Model training, evaluation, and hyperparameter tuning
- Model comparison and performance reporting
- Interactive web interface built with Streamlit  
- Real-time prediction from user input  

---

## 🧩 Dataset
The dataset contains various medical attributes such as:

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Maximum Heart Rate  
- Exercise-Induced Angina  
- ST Depression (oldpeak)  
- ST slope 
- Heart Disease  

> Dataset Source: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Joblib
- **Tools:** Jupyter Notebook
- **IDE:** VS Code

---

## 📈 Model Performance
| Model | Accuracy | F1-Score |
|--------|-----------|-----------|
| Logistic Regression | 0.87 | 0.88 |
| KNN | 0.86 | 0.88 |
| SVM | 0.85 | 0.87 |
| Naive Bayes | 0.85 | 0.86 |
| Decision Tree | 0.79 | 0.81 |

> Logistic Regression achieved the best overall performance with **87% accuracy**.

---

## 📂 Project Structure
1. The project structure I have:
   ```bash

   Heart-Disease-Prediction/
   │
   ├── DataSet/
   │ └── heart.csv
   ├── venv/
   ├── app.py
   ├── heart_disease_prediction.ipynb
   ├── columns.pkl
   ├── LR_Heart.pkl
   ├── Scaler.pkl
   ├── README.md
   └── requirements.txt


---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   
3. Create Virtual Environment:
    ```bash
    python -m venv venv
     
4. Run the Jupyter Notebook:

   After installing all the dependencies in the virtual environment, open the notebook and select the virtual environment as your kernel, select this virtual environment, and then run all the cells.     

## 📘 Future Improvements

- Integrate deep learning models for improved accuracy
- Connect to a database for storing patient predictions
- Improve UI with health insights and visualization dashboards

## 👨‍💻 Author
**Muhammad Talha**  
Final-year Computer Science student at UET Lahore  

📫 [Email](mailto:muhammadtalhashahid2005@gmail.com)  
🌐 [Portfolio](https://talhashahid.netlify.app)  
💼 [LinkedIn](https://www.linkedin.com/in/muhammadtaalhaa/)  
💻 [GitHub](https://github.com/RanaTalha04)
