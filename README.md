# 🩺 Diabetes Risk Predictor

A full-stack machine learning application that predicts a patient's risk of diabetes based on key health measurements. The project covers the complete ML lifecycle: data cleaning, model training and comparison, and deployment through a REST API with a custom web frontend.

## 🔗 Live Demo
[Add your deployed link here once hosted]

##  Problem Statement

Early identification of diabetes risk allows for earlier intervention and better health outcomes. This project uses the Pima Indians Diabetes Dataset to train a model that estimates diabetes risk from 8 routine clinical measurements, and exposes it through a simple web interface for real-time predictions.

## 🧠 Key Technical Highlights

- **Identified and corrected a hidden data quality issue**: several clinical columns (Glucose, Blood Pressure, BMI, Insulin, Skin Thickness) used `0` as a placeholder for missing values — a value that is medically impossible for these measurements. These were converted to proper missing values and imputed using the median (chosen over mean due to skewed distributions with outliers, e.g. Insulin: mean = 79.8 vs median = 30.5).
- **Compared three models** — Logistic Regression, Random Forest, and XGBoost — using accuracy, precision, recall, F1-score, and ROC-AUC, not accuracy alone.
- **Prioritized recall over raw accuracy**: in a medical screening context, missing an actual diabetic patient (false negative) is more costly than a false alarm, so `class_weight="balanced"` was used to reduce missed diagnoses.
- **Built a full-stack deployment**: a FastAPI backend serves the trained model via a REST API, with a custom HTML/CSS/JavaScript frontend making live prediction requests.

## 📊 Dataset

- **Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle / UCI)
- **Rows**: 768 patients
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target**: Outcome (0 = no diabetes, 1 = diabetes)

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| Modeling | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Backend / API | FastAPI, uvicorn |
| Frontend | HTML, CSS, JavaScript (vanilla, no framework) |
| Model persistence | joblib |

## 📈 Model Performance

| Model | Accuracy | Recall (Diabetic) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XX | 0.XX | 0.XX |
| XGBoost | 0.XX | 0.XX | 0.XX |

*(Fill in with your actual results from running `diabetes_project.py`)*

**Final model used**: [Logistic Regression / Random Forest / XGBoost] — chosen for [best recall / best ROC-AUC / best balance], since minimizing missed diabetic cases was prioritized over raw accuracy.

## 📁 Project Structure

ML_project/
├── diabetes_project.py # Data cleaning, training, and evaluation
├── diabetes.csv # Dataset
├── diabetes_model.pkl # Saved trained model + scaler
├── app.py # FastAPI backend serving the model
├── frontend/
│ └── index.html # Frontend UI
├── requirements.txt
└── README.md


## ▶️ How to Run Locally

1. **Clone the repository**
```bash
   git clone <your-repo-url>
   cd ML_project
```

2. **Create a virtual environment and install dependencies**
```bash
   python -m venv myenv
   myenv\Scripts\activate        # Windows
   pip install -r requirements.txt
```

3. **Train the model** (optional — a pre-trained model is already included)
```bash
   python diabetes_project.py
```

4. **Run the backend server**
```bash
   uvicorn app:app --reload
```

5. **Open the app**
   Visit `http://localhost:8000` in your browser.



## Disclaimer

This project is for educational purposes only and is not intended for actual medical diagnosis. Consult a healthcare professional for medical advice.
