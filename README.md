# Face Emotion Recognizer

This project is a Flask web application that performs emotion recognition from face images.
Features:
- Upload an image or use the webcam
- Detect faces using MTCNN
- Predict emotions with a Keras/TensorFlow model
- Save uploads to `uploads/` and `datasets/raw/` for later labeling

How to run:
1. Create a Python virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare training data under `datasets/labeled/<emotion>/` for each class (or place some images in datasets/raw for manual labeling).
4. Train model (optional): `python model_training.py`
5. Run app: `python app.py`
6. Open http://localhost:5000/

Notes:
- The repo includes a baseline model training script; you must provide labeled images to train.
- For production, secure file uploads and do not run training from the Flask process.
