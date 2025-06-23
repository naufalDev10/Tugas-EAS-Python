from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model dan label encoders
model = joblib.load("./models/diabetes_rf_model.pkl")
label_encoders = joblib.load("./models/label_encoders/label_encoders.pkl")

# Daftar fitur yang digunakan
features = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity", "Age", "Gender"
]

# Mapping nama fitur medis ke label ramah pengguna
feature_labels = {
    "Polyuria": "Sering buang air kecil",
    "Polydipsia": "Sering merasa haus",
    "sudden weight loss": "Penurunan berat badan secara tiba-tiba",
    "weakness": "Mudah lelah",
    "Polyphagia": "Sering lapar",
    "Genital thrush": "Infeksi jamur di area genital",
    "visual blurring": "Penglihatan kabur",
    "Itching": "Gatal-gatal",
    "Irritability": "Cepat marah",
    "delayed healing": "Luka sulit sembuh",
    "partial paresis": "Kelemahan sebagian otot",
    "muscle stiffness": "Otot kaku",
    "Alopecia": "Rambut rontok",
    "Obesity": "Kegemukan",
    "Age": "Usia",
    "Gender": "Jenis Kelamin"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = {}
        for feature in features:
            value = request.form.get(feature)
            input_data[feature] = value

        df = pd.DataFrame([input_data])

        # Encode kolom yang memerlukan encoder
        for column in df.columns:
            if column in label_encoders:
                le = label_encoders[column]
                df[column] = le.transform(df[column])

        # Atur ulang urutan kolom agar sesuai model
        df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]
        pred_label = label_encoders["class"].inverse_transform([prediction])[0]

        return render_template("index.html", result=pred_label, form_data=input_data, feature_labels=feature_labels)

    return render_template("index.html", result=None, form_data={}, feature_labels=feature_labels)
