from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

def evaluate_model(name, model, X_test, y_test, model_accuracies):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracies[name] = acc
    print(f"\n{name}", flush=True)
    print(f"Accuracy: {acc * 100:.2f}%", flush=True)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), flush=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred), flush=True)
    print("-" * 50, flush=True)

def train_and_select_model():
    print("Training models and selecting the best one...", flush=True)

    df = pd.read_csv("data/anemia.csv")
    df = df.drop(columns=['Name', 'Address', 'Phone'], errors='ignore')

    # Data Overview
    print("\n📊 Dataset Overview", flush=True)
    print(f"Shape: {df.shape}", flush=True)
    print("\nData Types:\n", df.dtypes, flush=True)
    print("\nFirst 5 Rows:\n", df.head(), flush=True)

    # Null Value Check
    print("\n🧼 Null Value Check", flush=True)
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.any() else "No missing values found.", flush=True)

    # Summary Statistics
    print("\n📈 Summary Statistics:\n", df.describe(), flush=True)

    # Encode categorical features if present
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"\n🔄 Encoding categorical columns: {cat_cols.tolist()}", flush=True)
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop('Result', axis=1)
    y = df['Result']

    global fields
    fields = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    model_accuracies = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    evaluate_model("Logistic Regression", lr, X_test_scaled, y_test, model_accuracies)

    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    evaluate_model("Random Forest", rf, X_test_scaled, y_test, model_accuracies)

    dt = DecisionTreeClassifier()
    dt.fit(X_train_scaled, y_train)
    evaluate_model("Decision Tree", dt, X_test_scaled, y_test, model_accuracies)

    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    evaluate_model("Naive Bayes", nb, X_test_scaled, y_test, model_accuracies)

    svm = SVC(probability=True)
    svm.fit(X_train_scaled, y_train)
    evaluate_model("SVM", svm, X_test_scaled, y_test, model_accuracies)

    gb = GradientBoostingClassifier()
    gb.fit(X_train_scaled, y_train)
    evaluate_model("Gradient Boosting", gb, X_test_scaled, y_test, model_accuracies)

    lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    lasso.fit(X_train_scaled, y_train)
    evaluate_model("Lasso (L1)", lasso, X_test_scaled, y_test, model_accuracies)

    best_model_name = max(model_accuracies, key=model_accuracies.get)
    all_models = {
        'Logistic Regression': lr,
        'Random Forest': rf,
        'Decision Tree': dt,
        'Naive Bayes': nb,
        'SVM': svm,
        'Gradient Boosting': gb,
        'Lasso (L1)': lasso
    }
    best_model = all_models[best_model_name]
    print(f"\n✅ Best Model Selected: {best_model_name}", flush=True)

    joblib.dump(best_model, "model.pkl")
    joblib.dump(model_accuracies, "model_accuracies.pkl")

    return best_model, scaler

if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    model, scaler = train_and_select_model()
else:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    fields = pd.read_csv("data/anemia.csv").drop(columns=['Result', 'Name', 'Address', 'Phone'], errors='ignore').columns.tolist()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['GET'])
def predict():
    return render_template("predict.html", fields=fields)

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/result', methods=['POST'])
def result():
    name = request.form['name']
    values = [float(request.form[field]) for field in fields]
    values_scaled = scaler.transform([values])
    prediction = model.predict(values_scaled)[0]
    result = "Anemic" if prediction == 1 else "Not Anemic"
    return render_template("result.html", name=name, result=result)

if __name__ == '__main__':
    print(">>> Starting Flask app...", flush=True)
    app.run(debug=True, use_reloader=False)
