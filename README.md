# 🌾 Crop Yield Prediction System using CNN & XGBoost

This mini-project presents a hybrid **AI-based system** for accurate **crop yield prediction** by integrating **image-based crop health analysis** with **weather data modeling**. It leverages **Convolutional Neural Networks (CNN)** for analyzing tomato crop images and **XGBoost** for processing environmental parameters, delivering reliable and actionable predictions for farmers.

---

## 📌 Project Title

**Crop Yield Prediction System by Integrating Image Analysis and Weather Data**

---

## 🎯 Objective

- Analyze crop health using **CNN** from leaf images (e.g., Early Blight, Late Blight)
- Predict yield using **XGBoost** based on weather data (temperature, humidity, etc.)
- Combine both predictions using **weighted averaging** for more robust yield forecasting
- Enable farmers to make data-driven, sustainable decisions

---

## 📦 Technologies Used

| Component            | Tool/Framework        |
|----------------------|------------------------|
| Programming Language | Python 3.8+            |
| Image Analysis       | TensorFlow, Keras (CNN)|
| Weather Modeling     | XGBoost                |
| Data Handling        | Pandas, NumPy          |
| Visualization        | Matplotlib             |
| Evaluation           | Scikit-learn (R², MAE, RMSE) |

---

## 📁 Datasets

1. **Crop Image Dataset**
   - Source: Open-source plant disease datasets
   - Classes: Healthy, Early Blight, Late Blight
   - Format: 224x224 images in separate class folders (train/test/val)

2. **Weather Dataset**
   - Format: CSV with temperature, humidity, RSSI, SNR, timestamp
   - Synthetic yield labels generated via logic-based mapping

---

## 🧠 Model Architecture

### 🖼️ CNN Model (for image classification)
- Input: Crop image (224x224)
- Layers: Conv → ReLU → Pool → FC → Softmax
- Output: Crop health class → mapped to a yield modifier (e.g., 0.1, 0.3)

### ☁️ XGBoost Model (for yield prediction)
- Input: Scaled weather features + derived time features
- Output: Estimated baseline yield (under ideal health)

### 🔗 Integration Formula
```python
Y_final = 0.4 * Y_CNN + 0.6 * Y_XGB
````

---

## ⚙️ How to Run

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/crop-yield-predictor.git
cd crop-yield-predictor
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Train models individually:

```bash
python train_cnn.py         # for CNN
python train_xgboost.py     # for XGBoost
```

4. Predict yield using integrated pipeline:

```bash
python predict_yield.py
```

---

## 📊 Results

| Model          | Accuracy | R² Score | MAE      | RMSE     |
| -------------- | -------- | -------- | -------- | -------- |
| CNN            | 58.71%   | —        | —        | —        |
| XGBoost        | —        | 0.87     | 0.02     | 0.57     |
| **Integrated** | **87%**  | **0.87** | **0.02** | **0.57** |

📌 *CNN model detects crop diseases, while XGBoost handles temporal weather variations. The integrated model offers superior accuracy.*

---

## 📈 Visualizations

* CNN accuracy/loss curve
* XGBoost feature importance (Temperature, RSSI most significant)
* Final yield prediction vs ground truth
* Comparison with Linear Regression & SVM

---

## ✅ Advantages

* Real-time insights into crop health and yield
* Combines environmental and visual data
* More accurate than traditional methods (e.g., linear regression, SVM)
* Scalable for future integration with IoT and other crops

---

## 📌 Future Work

* Add **soil** and **market data** for economic forecasting
* Use **multilabel image classification** for complex diseases
* Build a **web dashboard** or **mobile app** for farmers
* Integrate **real-time data** using IoT sensors and APIs

---
