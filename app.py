from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import numpy as np
import os
import sqlite3
import json
import re
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-before-deployment")
DATABASE = os.path.join(os.path.dirname(__file__), "patients.db")

data = load_breast_cancer()
X = data.data
y = data.target
FEATURE_NAMES = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

FEATURE_RANGES = {
    "mean radius":              {"min": 6.98,   "max": 28.11,   "mean": 14.13},
    "mean texture":             {"min": 9.71,   "max": 39.28,   "mean": 19.29},
    "mean perimeter":           {"min": 43.79,  "max": 188.50,  "mean": 91.97},
    "mean area":                {"min": 143.50, "max": 2501.00, "mean": 654.89},
    "mean smoothness":          {"min": 0.0526, "max": 0.1634,  "mean": 0.0964},
    "mean compactness":         {"min": 0.0194, "max": 0.3454,  "mean": 0.1043},
    "mean concavity":           {"min": 0.0,    "max": 0.4268,  "mean": 0.0888},
    "mean concave points":      {"min": 0.0,    "max": 0.2012,  "mean": 0.0489},
    "mean symmetry":            {"min": 0.1060, "max": 0.3040,  "mean": 0.1812},
    "mean fractal dimension":   {"min": 0.0500, "max": 0.0974,  "mean": 0.0628},
    "radius error":             {"min": 0.1115, "max": 2.873,   "mean": 0.4052},
    "texture error":            {"min": 0.3602, "max": 4.885,   "mean": 1.2169},
    "perimeter error":          {"min": 0.757,  "max": 21.98,   "mean": 2.8661},
    "area error":               {"min": 6.802,  "max": 542.20,  "mean": 40.337},
    "smoothness error":         {"min": 0.0017, "max": 0.0311,  "mean": 0.0070},
    "compactness error":        {"min": 0.0023, "max": 0.1354,  "mean": 0.0255},
    "concavity error":          {"min": 0.0,    "max": 0.3960,  "mean": 0.0319},
    "concave points error":     {"min": 0.0,    "max": 0.0528,  "mean": 0.0118},
    "symmetry error":           {"min": 0.0079, "max": 0.0790,  "mean": 0.0205},
    "fractal dimension error":  {"min": 0.0009, "max": 0.0298,  "mean": 0.0038},
    "worst radius":             {"min": 7.93,   "max": 36.04,   "mean": 16.27},
    "worst texture":            {"min": 12.02,  "max": 49.54,   "mean": 25.68},
    "worst perimeter":          {"min": 50.41,  "max": 251.20,  "mean": 107.26},
    "worst area":               {"min": 185.20, "max": 4254.00, "mean": 880.58},
    "worst smoothness":         {"min": 0.0712, "max": 0.2226,  "mean": 0.1324},
    "worst compactness":        {"min": 0.0273, "max": 1.0580,  "mean": 0.2543},
    "worst concavity":          {"min": 0.0,    "max": 1.2520,  "mean": 0.2722},
    "worst concave points":     {"min": 0.0,    "max": 0.2910,  "mean": 0.1146},
    "worst symmetry":           {"min": 0.1565, "max": 0.6638,  "mean": 0.2901},
    "worst fractal dimension":  {"min": 0.0550, "max": 0.2075,  "mean": 0.0839},
}

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE,
                full_name TEXT NOT NULL,
                mobile TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                result TEXT NOT NULL,
                benign_prob REAL NOT NULL,
                malignant_prob REAL NOT NULL,
                measurements TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
        """)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "patient_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


init_db()


@app.route("/")
@login_required
def index():
    groups = {
        "Mean Values": FEATURE_NAMES[:10],
        "Error Values": FEATURE_NAMES[10:20],
        "Worst Values": FEATURE_NAMES[20:],
    }
    return render_template(
        "index.html",
        groups=groups,
        ranges=FEATURE_RANGES,
        patient_name=session.get("patient_name", "Patient"),
        patient_id=session.get("patient_id", "")
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        mobile = request.form.get("mobile", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not all([full_name, mobile, email, password, confirm_password]):
            flash("Please fill in all fields.", "error")
            return render_template("register.html")
        if not re.fullmatch(r"[0-9]{10}", mobile):
            flash("Enter a valid 10-digit mobile number.", "error")
            return render_template("register.html")
        if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            flash("Enter a valid email address.", "error")
            return render_template("register.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return render_template("register.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        try:
            with get_db() as conn:
                cursor = conn.execute(
                    """INSERT INTO patients
                       (patient_id, full_name, mobile, email, password_hash)
                       VALUES (NULL, ?, ?, ?, ?)""",
                    (full_name, mobile, email, generate_password_hash(password))
                )
                new_patient_id = f"BCP{cursor.lastrowid:06d}"
                conn.execute(
                    "UPDATE patients SET patient_id = ? WHERE id = ?",
                    (new_patient_id, cursor.lastrowid)
                )
            session.clear()
            session["patient_id"] = new_patient_id
            session["patient_name"] = full_name
            flash(f"Registration successful! Your Patient ID is {new_patient_id}.", "success")
            return redirect(url_for("index"))
        except sqlite3.IntegrityError:
            flash("This email is already registered. Please log in.", "error")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            patient = conn.execute(
                "SELECT * FROM patients WHERE email = ?", (email,)
            ).fetchone()

        if patient and check_password_hash(patient["password_hash"], password):
            session.clear()
            session["patient_id"] = patient["patient_id"]
            session["patient_name"] = patient["full_name"]
            return redirect(url_for("index"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        records = conn.execute(
            """SELECT result, benign_prob, malignant_prob, created_at
               FROM predictions WHERE patient_id = ?
               ORDER BY id DESC""",
            (session["patient_id"],)
        ).fetchall()
    return render_template(
        "history.html",
        records=records,
        patient_name=session.get("patient_name", "Patient"),
        patient_id=session.get("patient_id", "")
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        features = [float(request.form.get(name, 0)) for name in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        label = "Benign" if prediction == 1 else "Malignant"
        benign_prob = round(float(probability[1]) * 100, 2)
        malignant_prob = round(float(probability[0]) * 100, 2)

        with get_db() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (patient_id, result, benign_prob, malignant_prob, measurements)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session["patient_id"],
                    label,
                    benign_prob,
                    malignant_prob,
                    json.dumps(dict(zip(FEATURE_NAMES, features)))
                )
            )

        return jsonify({
            "prediction": int(prediction),
            "label": label,
            "benign_prob": benign_prob,
            "malignant_prob": malignant_prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
