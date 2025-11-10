# ✈️ Aerospace Predictive Maintenance

### Week 2 - Edunet AI/ML Internship Project

**Objective:**  
Predict if an aircraft engine will fail within a particular cycle, using time-series sensor data.

---

## 📋 Steps Performed
1. Imported and preprocessed aircraft sensor data  
2. Generated Remaining Useful Life (RUL) feature  
3. Created binary target variable (failure within 30 cycles)  
4. Normalized data using StandardScaler  
5. Trained a Random Forest Classifier  
6. Evaluated model and saved results to `report_summary.json`

---

## 🧮 Model Summary
- Model: RandomForestClassifier  
- Accuracy: ~96.8%  
- Features: Sensor and cycle data  
- Type: Binary Classification (Failure/No Failure)

---

## ⚙️ Requirements
```
pip install -r requirements.txt
```

---

## ▶️ Run the Code
```
python main.py
```

---

## 📁 Files Included
- `main.py` → Source code  
- `requirements.txt` → Libraries  
- `report_summary.json` → Model summary  
- `aircraft_PM.csv` → Dataset  
