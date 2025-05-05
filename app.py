import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# ✅ Define categories (Ensure it matches your dataset)
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
              'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z']

# ✅ Configure upload folder
UPLOAD_FOLDER = 'static/uploaded'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Preprocess image
def preprocess_image(image_path):
    img_size = 128
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = img.reshape(1, img_size, img_size, 1)  # Reshape for model
    return img

# ✅ Homepage Route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    # ✅ Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # ✅ Preprocess & Predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predicted_sign = categories[predicted_label]

    # ✅ Render result page
    return render_template('result.html', filename=filename, prediction=predicted_sign)

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
