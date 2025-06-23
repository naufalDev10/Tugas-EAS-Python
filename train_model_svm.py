import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset
df = pd.read_csv("dataset/diabetes_data_upload.csv")

# 2. Preprocessing
data = df.copy()

# Ganti 'Yes'/'No' menjadi 1/0
binary_columns = data.columns.drop(['Age', 'Gender', 'class'])
data[binary_columns] = data[binary_columns].replace({'Yes': 1, 'No': 0})

# Label Encoding untuk kolom Gender dan class
le_gender = LabelEncoder()
le_class = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])  # Male=1, Female=0
data['class'] = le_class.fit_transform(data['class'])     # Positive=1, Negative=0

# 3. Pisahkan fitur dan target
X = data.drop('class', axis=1)
y = data['class']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Latih model SVM
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 6. Prediksi dan evaluasi
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

# 7. Tampilkan hasil
print("Akurasi:", accuracy)
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", matrix)
