import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("dataset/diabetes_data_upload.csv")

# 2. Data Preprocessing 
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# 3. Pisahkan fitur dan target
X = df.drop('class', axis=1)
y = df['class']

# Simpan urutan fitur
feature_order = X.columns.tolist()

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Buat dan latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Simpan model, encoder, dan urutan fitur
joblib.dump(model, "models/diabetes_rf_model.pkl")
joblib.dump(label_encoders, "models/label_encoders/label_encoders.pkl")
joblib.dump(feature_order, "models/feature_order/feature_order.pkl")
