# face_emotions.py
import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Emotion labels — adapt as needed
EMOTION_LABELS = ['angry','disgust','fear','happy','sad','surprise','neutral']

class FaceEmotionRecognizer:
    def __init__(self, model_path='models/emotion_model.h5', target_size=(48,48)):
        self.detector = MTCNN()
        self.model = None
        self.model_path = model_path
        self.target_size = target_size
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
            except Exception as e:
                print('Could not load model:', e)

    def load_model(self, path=None):
        path = path or self.model_path
        self.model = load_model(path)
        print(f"Loaded model from {path}")

    def detect_faces(self, image_np):
        faces = self.detector.detect_faces(image_np)
        return faces

    def preprocess_face(self, face_image):
        img = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, self.target_size)
        img = img.astype('float32')/255.0
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        return img

    def predict_emotion(self, face_image):
        if self.model is None:
            raise ValueError("Model not loaded.")
        x = self.preprocess_face(face_image)
        preds = self.model.predict(x)[0]
        idx = int(np.argmax(preds))
        return EMOTION_LABELS[idx], float(preds[idx]), preds

    def analyze_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        faces = self.detect_faces(img_np)
        results = []
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped = img_np[y:y+h, x:x+w]
            try:
                label, conf, probs = self.predict_emotion(cropped)
            except Exception as e:
                label, conf, probs = None, None, None
            results.append({
                'box': [int(x), int(y), int(w), int(h)],
                'label': label,
                'confidence': conf,
                'probs': None if probs is None else probs.tolist() if hasattr(probs, 'tolist') else None
            })
        return results
