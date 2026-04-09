# Depression Detection from Social Media using Hybrid Transformer Model

## 📌 Project Overview

This project focuses on detecting depression from social media text using a Hybrid Transformer Model. It combines the power of DistilBERT with handcrafted linguistic features to improve prediction accuracy and capture both contextual and statistical aspects of language.

## 🚀 Key Features

* Real-time depression prediction from user input text
* Hybrid model (DistilBERT + linguistic features)
* Confidence score and risk level classification (Low, Moderate, High)
* User authentication system (Login/Register)
* Prediction history tracking
* Admin dashboard for monitoring users and predictions
* Interactive web interface using Flask

## 🧠 Model Details

* Transformer Model: DistilBERT
* Additional Features:

  * Word count
  * Average word length
  * Negative word count
  * Pronoun usage
* Classifier: Fully connected neural network
* Loss Function: CrossEntropyLoss
* Optimizer: AdamW

## 📊 Dataset

The model was trained on a social media mental health dataset sourced from Kaggle.
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

## ⚙️ Tech Stack

* Backend: Python, Flask
* Frontend: HTML, CSS, Bootstrap, JavaScript
* Database: SQLite with SQLAlchemy
* Machine Learning: PyTorch, Hugging Face Transformers
* Visualization: Matplotlib, Chart.js

## 📁 Project Structure

* app/ – Flask backend and routes
* templates/ – HTML pages
* static/ – CSS, JS, charts
* model/ – Model architecture (trained model excluded)
* data/ – Dataset placeholder (excluded)
* requirements.txt – Dependencies

## ⚠️ Note

Due to file size limitations, the dataset and trained model files are not included in this repository. They can be added separately for full functionality.

## 🎯 Objective

To develop an intelligent system that can automatically detect depression from social media text and assist in early identification of mental health risks.

## 🔮 Future Scope

* Improve model accuracy with larger datasets
* Deploy using cloud platforms
* Add multilingual support
* Integrate with mobile applications
