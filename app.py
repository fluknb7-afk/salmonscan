import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # รวมไม่เกิน 32 MB ต่อคำขอ

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "salmonscan_pipeline.joblib"
model = joblib.load(MODEL_PATH)  # Pipeline = StandardScaler + SVM (ฟีเจอร์ Color histogram)


def extract_hist(img_rgb):
    """ต้องเหมือนตอนเทรนทุกขั้นตอน: resize 600x250 -> HSV -> histogram 8x8x8"""
    img_rgb = cv2.resize(img_rgb, (600, 250))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32).reshape(1, -1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"}), 400

    results = []
    for f in files:
        data = np.frombuffer(f.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            results.append({"name": f.filename, "error": "อ่านไฟล์ภาพไม่ได้"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        proba = model.predict_proba(extract_hist(rgb))[0]  # [Fresh, Infected]
        pred = int(np.argmax(proba))
        results.append({
            "name": f.filename,
            "label": "Infected" if pred == 1 else "Fresh",
            "p_fresh": float(proba[0]),
            "p_infected": float(proba[1]),
        })
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
