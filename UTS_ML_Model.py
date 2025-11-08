import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('student_lifestyle_dataset.csv')

# Encoding label Stress_Level
label_encoder = LabelEncoder()
df['Stress_Level'] = label_encoder.fit_transform(df['Stress_Level'])

# Pisahkan fitur dan label
X_classification = df.drop(['Student_ID', 'Stress_Level'], axis=1)
y_classification = df['Stress_Level']

# Split dataset
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classification, y_classification, 
    test_size=0.2, random_state=42
)

# Buat model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_clf, y_train_clf)

# Evaluasi
y_pred_clf = logistic_model.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
report = classification_report(y_test_clf, y_pred_clf)

print("Accuracy:", accuracy)
print(report)

# Simpan model & label encoder
joblib.dump(logistic_model, "stress_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model saved as stress_model.pkl")
print("Encoder saved as label_encoder.pkl")
