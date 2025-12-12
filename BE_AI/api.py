from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS untuk frontend

# Load model
try:
    with open("maternal_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None

# Simulasi database untuk history (gunakan database real untuk production)
history_data = []

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validasi input
        required_fields = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "message": f"Field {field} tidak ditemukan"
                }), 400

        age = float(data["Age"])
        sys = float(data["SystolicBP"])
        dias = float(data["DiastolicBP"])
        bs = float(data["BS"])
        temp = float(data["BodyTemp"])
        hr = float(data["HeartRate"])

        # Validasi range nilai
        if not (10 <= age <= 100):
            return jsonify({"success": False, "message": "Usia harus antara 10-100 tahun"}), 400
        if not (60 <= sys <= 200):
            return jsonify({"success": False, "message": "Systolic BP harus antara 60-200 mmHg"}), 400
        if not (40 <= dias <= 140):
            return jsonify({"success": False, "message": "Diastolic BP harus antara 40-140 mmHg"}), 400
        if not (1.0 <= bs <= 30.0):
            return jsonify({"success": False, "message": "Blood Sugar harus antara 1.0-30.0 mmol/L"}), 400
        if not (30 <= temp <= 45):
            return jsonify({"success": False, "message": "Suhu tubuh harus antara 30-45 °C"}), 400
        if not (40 <= hr <= 200):
            return jsonify({"success": False, "message": "Heart rate harus antara 40-200 bpm"}), 400

        # Prediksi
        if model is None:
            return jsonify({
                "success": False,
                "message": "Model belum dimuat"
            }), 500

        X = np.array([[age, sys, dias, bs, temp, hr]])
        pred = model.predict(X)[0]

        # Simpan ke history
        history_entry = {
            "id": len(history_data) + 1,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age": age,
            "systolicBP": sys,
            "diastolicBP": dias,
            "bs": bs,
            "bodyTemp": temp,
            "heartRate": hr,
            "risk": pred
        }
        history_data.append(history_entry)

        return jsonify({
            "success": True,
            "risk": pred,
            "message": "Prediksi berhasil"
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "message": f"Data tidak valid: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Terjadi kesalahan: {str(e)}"
        }), 500

@app.route("/history", methods=["GET"])
def get_history():
    try:
        # Return history sorted by newest first
        sorted_history = sorted(history_data, key=lambda x: x["date"], reverse=True)
        return jsonify({
            "success": True,
            "data": sorted_history
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Gagal mengambil history: {str(e)}"
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        if not history_data:
            return jsonify({
                "success": True,
                "data": {
                    "avgBloodPressure": "--",
                    "avgBloodSugar": "--",
                    "avgBMI": "--"
                }
            })

        # Hitung rata-rata 7 hari terakhir
        recent_data = history_data[-7:] if len(history_data) >= 7 else history_data

        avg_systolic = sum(d["systolicBP"] for d in recent_data) / len(recent_data)
        avg_diastolic = sum(d["diastolicBP"] for d in recent_data) / len(recent_data)
        avg_bs = sum(d["bs"] for d in recent_data) / len(recent_data)

        # Simplified BMI calculation (would need height data for accurate calculation)
        # Using placeholder for now
        avg_bmi = 22.5

        return jsonify({
            "success": True,
            "data": {
                "avgBloodPressure": f"{int(avg_systolic)}/{int(avg_diastolic)}",
                "avgBloodSugar": f"{int(avg_bs)}",
                "avgBMI": f"{avg_bmi:.1f}"
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Gagal mengambil statistik: {str(e)}"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "success": True,
        "message": "API is running",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    print("Starting Flask API server...")
    print("Server running at: http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
