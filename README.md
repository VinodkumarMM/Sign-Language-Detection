# Sign-Language-Detection
# 🧠 Sign Language Recognition Web App (Deep Learning + Flask)

This project is a web-based application for recognizing American Sign Language (ASL) alphabets (A-Z) and digits (0-9) using a trained Convolutional Neural Network (CNN). Users can upload hand gesture images, and the model predicts the corresponding sign.

---

## 📁 Directory Structure

sign_language_app/

│

├── static/

│ └── style.css # (Optional) Custom styling

│

├── templates/

│ ├── index.html # Upload page

│ └── result.html # Prediction result page

│

├── uploads/ # Stores uploaded images

│

├── sign_language_model.h5 # Pre-trained CNN model

├── app.py # Main Flask application

├── requirements.txt # Python dependencies

├── README.md # Project documentation



## 🚀 Features

- Recognizes 36 ASL gestures (0-9 and A-Z).
- Upload any hand gesture image.
- Predicts and displays the result.
- Option to upload another image for prediction.

---

## 🧠 Technologies Used

- Python 3
- TensorFlow / Keras
- OpenCV
- Flask
- HTML5 / CSS3
- NumPy

---

## ✅ Setup Instructions

# Download the Pretrained model
from here : https://drive.google.com/file/d/1W4R99ergEjgcMNJR-QMcJDmuJ8xGF8kv/view?usp=drive_link

# 📷 Usage
On the home page, upload a grayscale image of a hand showing a gesture.

The model predicts and displays the sign.

Click "Upload Another Image" to predict again.

# 📈 Future Enhancements
Add real-time webcam support.

Support for Indian Sign Language (ISL).

Incorporate NLP for sentence translation.

Responsive mobile UI.

# 🙏 Credits
Dataset: Custom ASL image dataset.

Libraries: TensorFlow, Keras, Flask, OpenCV.

# 📄 License
This project is licensed under the MIT License. Feel free to use and modify it for educational purposes.
