# link_app.py
import os
import base64
from datetime import datetime
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
DATASET_RAW = 'datasets/raw'
ALLOWED_EXT = {'png','jpg','jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_RAW, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def save_upload_file(file_storage):
    filename = secure_filename(file_storage.filename)
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    save_name = f"{ts}_{filename}"
    path = os.path.join(UPLOAD_FOLDER, save_name)
    file_storage.save(path)
    raw_path = os.path.join(DATASET_RAW, save_name)
    with open(path, 'rb') as fr, open(raw_path, 'wb') as fw:
        fw.write(fr.read())
    return path

def save_base64_image(b64_string):
    header, encoded = b64_string.split(',',1)
    ext = 'png' if 'png' in header else 'jpg'
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    filename = f"{ts}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    with open(path, 'wb') as f:
        f.write(base64.b64decode(encoded))
    raw_path = os.path.join(DATASET_RAW, filename)
    with open(raw_path, 'wb') as fw, open(path, 'rb') as fr:
        fw.write(fr.read())
    return path
