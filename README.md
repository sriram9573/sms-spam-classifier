# SMS Spam Classifier 📱 🚫

**AI-Powered Message Filtering with NLP**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)](https://www.nltk.org/)

---

## 🚀 Overview

SMS Spam Classifier is a machine learning web application that detects spam messages with high accuracy using Natural Language Processing (NLP) techniques. Built with Streamlit, it provides an intuitive interface where users can paste any SMS or email text and instantly receive a spam/not-spam prediction.

The model leverages **TF-IDF vectorization** and a trained classification algorithm to analyze message patterns, making it effective at identifying phishing attempts, promotional spam, and fraudulent messages.

---

## 📖 About

**The Problem**: Spam messages clutter our inboxes, waste time, and pose security risks through phishing and scams. Manual filtering is tedious, and simple keyword-based filters are easily bypassed by sophisticated spam techniques.

**The Solution**: This classifier uses machine learning trained on thousands of real SMS messages to identify spam with high precision. By analyzing linguistic patterns, word frequencies, and message structure, it can detect spam even when traditional filters fail.

**The Vision**: To provide an accessible, lightweight spam detection tool that anyone can deploy locally or in the cloud. Whether you're building a messaging app, email client, or just want to test suspicious messages, this classifier offers production-ready spam detection.

---

## ✨ Key Features

### 🤖 **Machine Learning Classification**
- **Pre-trained Model**: Ready-to-use classifier trained on real SMS spam dataset.
- **High Accuracy**: Achieves strong performance on spam detection tasks.
- **Binary Classification**: Clear spam/not-spam predictions.

### 🔤 **Advanced NLP Pipeline**
- **Text Preprocessing**: Lowercasing, tokenization, and punctuation removal.
- **Stopword Filtering**: Removes common words that don't contribute to spam detection.
- **Stemming**: Reduces words to their root form using Porter Stemmer.
- **TF-IDF Vectorization**: Converts text to numerical features for ML model.

### 🌐 **Interactive Web Interface**
- **Streamlit UI**: Clean, responsive interface built with Streamlit.
- **Real-time Predictions**: Instant classification results.
- **Easy Deployment**: Can be deployed to Streamlit Cloud, Heroku, or any cloud platform.

---

## 🛠️ Technical Stack

### Machine Learning
- **Algorithm**: Classification model (stored in `model.pkl`)
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Library**: scikit-learn

### Natural Language Processing
- **Tokenization**: NLTK word tokenizer
- **Stemming**: Porter Stemmer
- **Stopwords**: NLTK English stopwords corpus

### Web Framework
- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Deployment**: Configured for Heroku (Procfile, setup.sh)

---

## 🏗️ Architecture

The application follows a simple yet effective pipeline:

1. **User Input**: User enters SMS/email text in the Streamlit interface
2. **Preprocessing**: Text is cleaned, tokenized, and stemmed
3. **Vectorization**: Processed text is converted to TF-IDF features
4. **Prediction**: ML model classifies the message as spam or not spam
5. **Display**: Result is shown to the user in real-time

---

## 🏁 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/sriram9573/sms-spam-classifier.git
    cd sms-spam-classifier
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data**
    ```python
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```

4.  **Run the application**
    ```bash
    streamlit run app.py
    ```

5.  **Access the app**
    
    Open your browser and navigate to `http://localhost:8501`

---

## 🎯 Usage

1. **Launch the app** using `streamlit run app.py`
2. **Enter a message** in the text area (SMS or email content)
3. **Click "Predict"** to classify the message
4. **View the result**: The app will display "Spam" or "Not Spam"

### Example Messages to Test:

**Spam:**
```
URGENT! You've won a $1000 Walmart gift card. Click here to claim: http://bit.ly/scam123
```

**Not Spam:**
```
Hey, are we still meeting for lunch at 1pm today?
```

---

## 📊 Model Details

- **Training Data**: `spam.csv` - SMS Spam Collection dataset
- **Features**: TF-IDF vectors extracted from preprocessed text
- **Model**: Pre-trained classifier (stored in `model.pkl`)
- **Vectorizer**: Pre-fitted TF-IDF vectorizer (stored in `vectorizer.pkl`)

The Jupyter notebook `sms-spam-detection.ipynb` contains the full training pipeline and model evaluation metrics.

---

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Deploy!

### Deploy to Heroku

The repository includes Heroku configuration files:
- `Procfile`: Defines the web process
- `setup.sh`: Streamlit configuration script

```bash
heroku create your-app-name
git push heroku main
```

---

## 🚀 Future Roadmap
- [ ] Multi-language support
- [ ] Confidence score display
- [ ] Model retraining interface
- [ ] API endpoint for programmatic access
- [ ] Browser extension integration
- [ ] Real-time email client integration

---


