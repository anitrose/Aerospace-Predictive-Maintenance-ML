# ✈️ Aerospace Predictive Maintenance using Machine Learning
> 🚀 Built as part of Edunet AI/ML Aerospace Internship — applying machine learning to real-world predictive maintenance problems.

## 📌 Overview
This project focuses on **predictive maintenance in aerospace systems** using the NASA CMAPSS dataset.  
The objective is to predict the **Remaining Useful Life (RUL)** of aircraft engines to enable proactive maintenance, improve safety, and reduce operational costs.

---


## 🎯 Problem Statement
Aircraft engine failures can lead to high costs and safety risks.  
This project builds a machine learning model to **predict engine degradation over time** and estimate how long an engine can operate before failure.

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Cleaned and structured CMAPSS dataset  
   - Handled missing values and normalized features  

2. **Feature Engineering**
   - Selected relevant sensor features  
   - Generated useful indicators for engine health  

3. **Exploratory Data Analysis (EDA)**
   - Visualized sensor behavior over time  
   - Identified patterns related to engine degradation  

4. **Model Building**
   - Trained machine learning models to predict RUL  
   - Evaluated performance using appropriate metrics  

5. **Evaluation**
   - Compared model predictions with actual RUL values  
   - Measured performance using metrics like RMSE  

---

## 📊 Key Insights
- Sensor readings such as **temperature, pressure, and vibration** show strong correlation with engine failure  
- Engine degradation follows identifiable patterns over time  
- Machine learning models can effectively estimate Remaining Useful Life  

---

## 🚀 Results
- Achieved **~96% accuracy** using a Random Forest classifier  
- Successfully predicted aircraft engine failure within **30 operational cycles**  
- Enabled early detection of failures, supporting proactive maintenance and reducing downtime  

---

## 📂 Project Structure
Aerospace-Predictive-Maintenance-ML/
│── data/
│── notebooks/
│── models/
│── README.md


---

## 💡 Future Improvements
- Use advanced models (XGBoost, LSTM for time-series)
- Deploy as a web dashboard for real-time monitoring
- Integrate live sensor data simulation

---
## 📊 Results & Visualizations

### 📊 RUL Distribution
![RUL Distribution](https://github.com/user-attachments/assets/b00c45ee-27b8-4d8f-b4ac-02538a6ce045)

Most engines operate close to failure (low RUL), indicating realistic degradation patterns in the dataset.

---

### 🔥 Correlation Heatmap
![Heatmap](https://github.com/user-attachments/assets/cd4d79c8-c542-4494-84db-ccf705923f6f)

Several sensors show strong positive and negative correlations, helping identify the most influential features for failure prediction.

---

### 📈 Sensor Behavior Over Time
![Sensor](https://github.com/user-attachments/assets/22f4b90b-aea9-42c9-b313-296d9c271550)

Sensor values change over time, revealing clear degradation trends as the engine approaches failure.

---

### 🤖 Model Performance
- **Model:** Random Forest Classifier  
- **Accuracy:** ~96%  

The model effectively predicts engine failure within 30 cycles, enabling proactive maintenance decisions.

## 🧠 Key Learning
- Learned how to handle time-series sensor data
- Understood importance of feature selection in ML models
- Gained experience in building real-world predictive systems


## 📎 Conclusion
This project demonstrates how **data science and machine learning can be applied in aerospace engineering** to improve safety, reduce downtime, and enable intelligent maintenance systems.
