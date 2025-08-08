# Diabetes-Prediction-Model

This project uses a **Gaussian Naive Bayes** classifier to predict whether a patient has diabetes based on two key features: **glucose level** and **blood pressure**.

## 📊 Dataset
The dataset is a CSV file containing:
- `glucose`: Patient's glucose level
- `bloodpressure`: Patient's blood pressure
- `diabetes`: Target variable (0 = No diabetes, 1 = Diabetes)

## ⚙️ Model
- **Algorithm**: Gaussian Naive Bayes (`GaussianNB`)
- **Library**: scikit-learn
- **Task**: Binary classification

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🧪 How to Run
1. Clone the repository:
   git clone https://github.com/AK-Jeevan/Diabetes-Prediction-Model.git
   
Install dependencies:
pip install -r requirements.txt

Run the script:
python diabetes_predictor.py 

📌 Highlights
Simple and interpretable model

Fast training and prediction

Great for healthcare ML beginners

📁 Files
Diabetes.csv: Dataset

diabetes_predictor.py: Model training and evaluation

README.md: Project overview

📜 License
This project is open-source and free to use under the MIT License.

Feel free to fork, star ⭐, or contribute!
