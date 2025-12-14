from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "maternal_model.pkl")

print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] MODEL_PATH: {MODEL_PATH}")
print(f"[DEBUG] Model file exists: {os.path.exists(MODEL_PATH)}")

model = None

try:
    print(f"[INFO] Attempting to load model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[OK] Model loaded successfully: {type(model)}")
    print(f"[OK] Model has predict method: {hasattr(model, 'predict')}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()

history_data = []

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json


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

     
        if not (10 <= age <= 100):
            return jsonify({"success": False, "message": "Usia harus antara 10-100 tahun"}), 400
        if not (60 <= sys <= 200):
            return jsonify({"success": False, "message": "Systolic BP harus antara 60-200 mmHg"}), 400
        if not (40 <= dias <= 140):
            return jsonify({"success": False, "message": "Diastolic BP harus antara 40-140 mmHg"}), 400
        if not (1.0 <= bs <= 30.0):
            return jsonify({"success": False, "message": "Blood Sugar harus antara 1.0-30.0 mmol/L"}), 400
        if not (35 <= temp <= 42):
            return jsonify({"success": False, "message": "Suhu tubuh harus antara 35-42 °C"}), 400
        if not (40 <= hr <= 200):
            return jsonify({"success": False, "message": "Heart rate harus antara 40-200 bpm"}), 400


        if model is None:
            return jsonify({
                "success": False,
                "message": "Model belum dimuat"
            }), 500

        X = np.array([[age, sys, dias, bs, temp, hr]])
        pred = model.predict(X)[0]

        #
        warnings = []
        recommendations = []

        if pred == "high risk":
            warnings.append(" PERINGATAN: Risiko kesehatan maternal tinggi terdeteksi!")
            recommendations.append(" Segera konsultasikan dengan dokter kandungan Anda")
            recommendations.append(" Jangan tunda untuk mendapatkan pemeriksaan medis")
            recommendations.append(" Pertimbangkan untuk segera ke rumah sakit jika mengalami gejala tidak biasa")
        elif pred == "mid risk":
            warnings.append(" Perhatian: Risiko kesehatan maternal sedang")
            recommendations.append(" Jadwalkan konsultasi dengan dokter dalam waktu dekat")
            recommendations.append(" Monitor kondisi kesehatan Anda secara teratur")
            recommendations.append(" Ikuti saran medis yang telah diberikan")
        else:
            recommendations.append(" Pertahankan pola hidup sehat")
            recommendations.append(" Rutin melakukan pemeriksaan kesehatan")
            recommendations.append(" Jaga pola makan bergizi seimbang")

        
        if sys > 140 or dias > 90:
            warnings.append(" Tekanan darah tinggi terdeteksi (Hipertensi)")
            recommendations.append(" Konsultasi dengan dokter tentang tekanan darah Anda")
        elif sys < 90 or dias < 60:
            warnings.append(" Tekanan darah rendah terdeteksi (Hipotensi)")

        if bs > 11.0:
            warnings.append(" Kadar gula darah tinggi terdeteksi")
            recommendations.append(" Batasi konsumsi gula dan karbohidrat sederhana")
        elif bs < 3.9:
            warnings.append(" Kadar gula darah rendah terdeteksi")

        if temp > 37.5:
            warnings.append(" Demam atau suhu tubuh tinggi terdeteksi")
            recommendations.append(" Monitor suhu tubuh secara berkala")
        elif temp < 36.1:
            warnings.append(" Suhu tubuh rendah terdeteksi (Hipotermia)")

        if hr > 100:
            warnings.append(" Detak jantung cepat terdeteksi (Takikardia)")
        elif hr < 60:
            warnings.append(" Detak jantung lambat terdeteksi (Bradikardia)")

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
            "message": "Prediksi berhasil",
            "warnings": warnings,
            "recommendations": recommendations
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

        recent_data = history_data[-7:] if len(history_data) >= 7 else history_data

        avg_systolic = sum(d["systolicBP"] for d in recent_data) / len(recent_data)
        avg_diastolic = sum(d["diastolicBP"] for d in recent_data) / len(recent_data)
        avg_bs = sum(d["bs"] for d in recent_data) / len(recent_data)

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
    app.run(port=5000, debug=True, use_reloader=False)
