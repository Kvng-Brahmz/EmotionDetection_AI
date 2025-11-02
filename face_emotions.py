import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image

MODEL_PATH = "models/emotion_model.h5"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
LABELS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

face_detector = cv2.CascadeClassifier(HAAR_PATH)

_model = None
def load_emotion_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing {MODEL_PATH}. Please ensure the pretrained model is downloaded.")
        _model = load_model(MODEL_PATH)
    return _model

def detect_faces(gray):
    return face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

def preprocess_face(face_gray, size=(48,48)):
    face = cv2.resize(face_gray, size)
    face = face.astype("float")/255.0
    face = img_to_array(face)
    return np.expand_dims(face, axis=0)

def predict_emotions_from_cv2_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    model = load_emotion_model()
    results = []
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        inp = preprocess_face(face)
        preds = model.predict(inp, verbose=0)[0]
        label = LABELS[np.argmax(preds)]
        conf = float(np.max(preds))
        results.append({"box":[int(x),int(y),int(w),int(h)],"label":label,"confidence":conf})
    if not results:
        return [{"label":"no_face_detected","confidence":0.0}]
    return results

def pil_bytes_to_cv2(b):
    img = Image.open(BytesIO(b)).convert("RGB")
    arr = np.array(img)[:,:,::-1]
    return arr

def save_image_bytes(b, prefix="uploaded"):
    Path("datasets").mkdir(exist_ok=True)
    fn = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
    p = os.path.join("datasets", fn)
    open(p, "wb").write(b)
    return p

def predict_from_image_bytes(b, save_image=False):
    img = pil_bytes_to_cv2(b)
    if save_image:
        save_image_bytes(b)
    return predict_emotions_from_cv2_image(img)
