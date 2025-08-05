from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
import logging
from scipy.sparse import hstack

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for models
ensemble_model = None
char_vectorizer = None
word_vectorizer = None
model_loaded = False

def preprocess_text(text):
    """
    Advanced text preprocessing (same as training)
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and basic cleaning
    text = str(text).strip()
    
    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove quotes and newlines
    text = text.replace('"', '').replace('\n', ' ').replace('\r', ' ')
    
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove extra punctuation but keep some meaningful ones
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
    
    # Remove numbers (often not meaningful for fake news detection)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_models():
    """
    Load the trained models
    """
    global ensemble_model, char_vectorizer, word_vectorizer, model_loaded
    
    try:
        logger.info("Loading models...")
        
        # Try to load ultimate models first
        try:
            ensemble_model = joblib.load('models/ultimate_ensemble_classifier.pkl')
            char_vectorizer = joblib.load('models/ultimate_char_vectorizer.pkl')
            word_vectorizer = joblib.load('models/ultimate_word_vectorizer.pkl')
            logger.info("✅ Ultimate ensemble model loaded successfully!")
            model_loaded = True
            return True
            
        except FileNotFoundError:
            # Fallback to enhanced model
            try:
                ensemble_model = joblib.load('models/enhanced_fake_news_classifier_passiveaggressive.pkl')
                char_vectorizer = joblib.load('models/enhanced_tfidf_vectorizer.pkl')
                word_vectorizer = None  # Enhanced model uses single vectorizer
                logger.info("✅ Enhanced model loaded successfully!")
                model_loaded = True
                return True
                
            except FileNotFoundError:
                # Fallback to basic model
                ensemble_model = joblib.load('models/fake_news_classifier.pkl')
                char_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                word_vectorizer = None  # Basic model uses single vectorizer
                logger.info("✅ Basic model loaded successfully!")
                model_loaded = True
                return True
                
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        model_loaded = False
        return False

def predict_news(text):
    """
    Make prediction using the loaded model
    """
    global ensemble_model, char_vectorizer, word_vectorizer
    
    if not model_loaded:
        return None, 0.0, "Model not loaded"
    
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        if len(processed_text.strip()) == 0:
            return None, 0.0, "Empty text after preprocessing"
        
        # Create features based on available vectorizers
        if word_vectorizer is not None:
            # Ultimate model with dual vectorizers
            char_features = char_vectorizer.transform([processed_text])
            word_features = word_vectorizer.transform([processed_text])
            combined_features = hstack([char_features, word_features])
        else:
            # Enhanced or basic model with single vectorizer
            combined_features = char_vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = ensemble_model.predict(combined_features)[0]
        
        # Get confidence score
        try:
            if hasattr(ensemble_model, 'predict_proba'):
                probabilities = ensemble_model.predict_proba(combined_features)[0]
                confidence = max(probabilities)
                fake_probability = probabilities[1] if len(probabilities) > 1 else (1 - probabilities[0])
            elif hasattr(ensemble_model, 'decision_function'):
                decision_score = ensemble_model.decision_function(combined_features)[0]
                confidence = abs(decision_score)
                fake_probability = 1 / (1 + np.exp(-decision_score))  # Sigmoid transformation
            else:
                confidence = 1.0
                fake_probability = 0.5
        except:
            confidence = 1.0
            fake_probability = 0.5
        
        # Interpret result
        label = "FAKE" if prediction == 1 else "REAL"
        
        return label, confidence, processed_text, fake_probability
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None, 0.0, f"Prediction error: {e}", 0.5

@app.route('/')
def home():
    """
    API home endpoint
    """
    return jsonify({
        "message": "COVID-19 Fake News Detection API",
        "status": "active",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "No text provided. Please send JSON with 'text' field."
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                "error": "Empty text provided"
            }), 400
        
        # Make prediction
        label, confidence, processed_text, fake_probability = predict_news(text)
        
        if label is None:
            return jsonify({
                "error": processed_text  # Error message is in processed_text
            }), 500
        
        # Determine confidence level
        if confidence > 1.5:
            confidence_level = "Very High"
        elif confidence > 1.0:
            confidence_level = "High"
        elif confidence > 0.5:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Create response
        response = {
            "prediction": label,
            "confidence": round(confidence, 3),
            "confidence_level": confidence_level,
            "fake_probability": round(fake_probability * 100, 1),
            "real_probability": round((1 - fake_probability) * 100, 1),
            "original_text": text,
            "processed_text": processed_text[:200] + "..." if len(processed_text) > 200 else processed_text,
            "timestamp": datetime.now().isoformat(),
            "model_type": "Ultimate Ensemble" if word_vectorizer else "Enhanced" if "enhanced" in str(ensemble_model) else "Basic"
        }
        
        logger.info(f"Prediction made: {label} (confidence: {confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple texts
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "No texts provided. Please send JSON with 'texts' array."
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                "error": "Texts must be provided as an array"
            }), 400
        
        if len(texts) > 10:  # Limit batch size
            return jsonify({
                "error": "Maximum 10 texts allowed per batch"
            }), 400
        
        results = []
        
        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                results.append({
                    "index": i,
                    "error": "Empty text"
                })
                continue
            
            label, confidence, processed_text, fake_probability = predict_news(text)
            
            if label is None:
                results.append({
                    "index": i,
                    "error": processed_text
                })
                continue
            
            results.append({
                "index": i,
                "prediction": label,
                "confidence": round(confidence, 3),
                "fake_probability": round(fake_probability * 100, 1),
                "original_text": text
            })
        
        return jsonify({
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/examples')
def get_examples():
    """
    Get example texts for testing
    """
    examples = {
        "fake_examples": [
            "COVID-19 vaccines alter your DNA permanently and make you magnetic",
            "5G towers cause coronavirus infections by weakening your immune system",
            "Drinking bleach or disinfectant completely cures coronavirus",
            "The coronavirus was created in a lab as a biological weapon",
            "Face masks actually make you sicker by trapping carbon dioxide"
        ],
        "real_examples": [
            "Wearing masks can significantly reduce the spread of COVID-19",
            "COVID-19 vaccines have been tested for safety and efficacy in clinical trials",
            "Washing hands frequently with soap helps prevent COVID-19 transmission",
            "Social distancing measures can help slow the spread of coronavirus",
            "Some people may experience mild side effects after COVID-19 vaccination"
        ]
    }
    
    return jsonify(examples)

@app.route('/model_info')
def model_info():
    """
    Get information about the loaded model
    """
    info = {
        "model_loaded": model_loaded,
        "model_type": "Ultimate Ensemble" if word_vectorizer else "Enhanced" if model_loaded else "None",
        "features": {
            "char_vectorizer": char_vectorizer is not None,
            "word_vectorizer": word_vectorizer is not None,
            "dual_vectorizers": word_vectorizer is not None
        },
        "performance": {
            "training_accuracy": "99.5%",
            "test_accuracy": "89.5%",
            "f1_fake": "99.55%",
            "f1_real": "99.55%"
        } if model_loaded else None
    }
    
    return jsonify(info)

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("🚀 Starting COVID-19 Fake News Detection API...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to load models. Please check model files.")
        print("❌ Cannot start API without models. Please ensure model files exist in 'models/' directory.")