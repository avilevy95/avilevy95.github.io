import re
import uuid
import os
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import gdown

# Directories for image classification
NEW_DATA_DIR = 'new_data'
IN_PROCESS_DIR = 'in_process'
CLASSIFIED_DIR = 'classified'
GOOD_DIR = os.path.join(CLASSIFIED_DIR, 'good')
BAD_DIR = os.path.join(CLASSIFIED_DIR, 'bad')

# Create necessary directories
for directory in [NEW_DATA_DIR, IN_PROCESS_DIR, GOOD_DIR, BAD_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

app = Flask(__name__)
app.secret_key = 'mzp.ta'  # Secret key for session management
CORS(app)

THRESHOLD = 0.41
MODEL_PATH = "model.h5"
ADMIN_SUFFIX = r'.*@mzp.ta$'
SERVER_IP = "https://f61c-37-60-47-6.ngrok-free.app"

def download_model():
    url = 'https://drive.google.com/uc?id=15AT9oF7jSNHYOEhe92uR_WbWgPjzLx8i'
    output = MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(url, output, quiet=False, verify=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists, skipping download.")

# Call the function to download the model
download_model()

class ModelHandler:
    def __init__(self, model_path, threshold):
        self.model = load_model(model_path)
        self.threshold = threshold

    def classify_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        prediction = self.model.predict(img)[0]
        predicted_class = "good" if prediction > self.threshold else "bad"
        return predicted_class

model_handler = ModelHandler(MODEL_PATH, THRESHOLD)

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message': 'Welcome to classifying fingerprints.'})

@app.route('/<filename>')
def uploaded_file(filename):

    # if not session.get('logged_in'):
    #     return jsonify({'error': 'Unauthorized access'}), 401
    return send_from_directory(IN_PROCESS_DIR, filename)

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(NEW_DATA_DIR, unique_filename)
        file.save(file_path)
        print(file_path)

        predicted_class = model_handler.classify_image(file_path)


        return jsonify({'class': predicted_class, 'file_name': unique_filename})

@app.route('/get-images', methods=['GET'])
def get_images():
    # if not session.get('logged_in'):
    #     return jsonify({'error': 'Unauthorized access'}), 401

    images = []
    for filename in os.listdir(NEW_DATA_DIR):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            images.append({
                'uri': f'{SERVER_IP}/{filename}'
            })
            # Move file to in_process for admin review
            in_process_path = os.path.join(IN_PROCESS_DIR, filename)
            os.rename(f'{NEW_DATA_DIR}/{filename}', in_process_path)
    print(images)
    return jsonify({'images': images})

@app.route('/classify-image', methods=['POST'])
def set_classify_image():
    # if not session.get('logged_in'):
    #     return jsonify({'error': 'Unauthorized access'}), 500

    data = request.get_json()

    image_name = os.path.basename(data['image']['uri'])
    print(image_name)
    classification = data['classification']
    image_path = os.path.join(IN_PROCESS_DIR, image_name)

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    if classification == 'good':
        destination_dir = GOOD_DIR
    else:
        destination_dir = BAD_DIR

    destination_path = os.path.join(destination_dir, image_name)
    os.rename(image_path, destination_path)

    return jsonify({'image': image_name, 'classification': classification})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    password = data.get('password')
    if not password:
        return jsonify({'message': 'Password is required'}), 500

    if re.match(ADMIN_SUFFIX, password):
        session['logged_in'] = True
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'message': 'Invalid credentials'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('logged_in', None)
    return jsonify({'message': 'Logout successful'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
