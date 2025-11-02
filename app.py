import os
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from link_app import predict_from_image_bytes, save_incoming_image
from pathlib import Path

UPLOAD_FOLDER = "datasets"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = MODEL_DIR / "emotion_model.h5"

# Pretrained model download URL (if missing, app will try to download at runtime)
PRETRAINED_URL = "https://github.com/openai-project-assets/emotion-models/releases/download/v1/emotion_model_small.h5"
if not MODEL_PATH.exists():
    try:
        print("Downloading pretrained emotion model...")
        r = requests.get(PRETRAINED_URL, timeout=30)
        r.raise_for_status()
        MODEL_PATH.write_bytes(r.content)
        print(f"Pretrained model saved at {MODEL_PATH}")
    except Exception as e:
        print("Could not download pretrained model automatically. You can place a Keras .h5 model at models/emotion_model.h5")
        print("Download error:", e)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

@app.route("/")
def index():
    return render_template("index.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "empty filename"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        with open(save_path, "rb") as f:
            img_bytes = f.read()
        result = predict_from_image_bytes(img_bytes, save_image=False)
        return jsonify({"success": True, "result": result})
    return jsonify({"success": False, "error": "invalid file type"}), 400

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_data()
    if not data:
        return jsonify({"success": False, "error": "no data"}), 400
    saved_path = save_incoming_image(data)
    result = predict_from_image_bytes(data, save_image=False)
    return jsonify({"success": True, "result": result, "saved_as": saved_path})

@app.route("/datasets/<path:filename>")
def datasets_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
