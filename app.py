# app.py
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from face_emotions import FaceEmotionRecognizer
import link_app

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = link_app.UPLOAD_FOLDER

# instantiate recognizer (loads model if exists)
recognizer = FaceEmotionRecognizer(model_path='models/emotion_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    if 'file' not in request.files:
        return jsonify({'error':'no file part'}), 400
    file = request.files['file']
    if file and link_app.allowed_file(file.filename):
        path = link_app.save_upload_file(file)
        results = recognizer.analyze_image(path)
        return jsonify({'image': path, 'results': results})
    else:
        return jsonify({'error':'invalid file'}), 400

@app.route('/analyze_webcam', methods=['POST'])
def analyze_webcam():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error':'no image data'}), 400
    b64 = data['image']
    path = link_app.save_base64_image(b64)
    results = recognizer.analyze_image(path)
    return jsonify({'image': path, 'results': results})

@app.route('/retrain', methods=['POST'])
def retrain_route():
    # Trigger retraining by launching model_training.py as a subprocess.
    # In production, prefer a background worker. Here we start the process and return.
    import subprocess
    subprocess.Popen(['python','model_training.py'])
    return jsonify({'status':'retrain started'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
