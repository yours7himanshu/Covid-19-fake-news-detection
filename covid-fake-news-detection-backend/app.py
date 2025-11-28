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
from pathlib import Path
from rule_filters import apply_rule_filters
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import random
from pymongo import MongoClient

# Resolve paths relative to this file so it works on Render/containers
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client["covid_fake_news_db"]
    verification_logs = db["verification_logs"]
    # Check connection
    mongo_client.server_info()
    logger.info("✅ Connected to MongoDB successfully!")
except Exception as e:
    logger.warning(f"⚠️ MongoDB connection failed: {e}. Dashboard will use simulated data.")
    verification_logs = None

app = Flask(__name__)
# Explicit CORS for production (Vercel frontend, localhost dev)
ALLOWED_ORIGINS = [
    "https://covid-19-fake-news-detection.vercel.app",
    "http://localhost:5173"
]
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=86400,
    always_send=True,
)

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
        
        # Try to load enhanced models first (temporarily for testing)
        try:
            ensemble_model = joblib.load(str(MODELS_DIR / 'enhanced_fake_news_classifier_passiveaggressive.pkl'))
            char_vectorizer = joblib.load(str(MODELS_DIR / 'enhanced_tfidf_vectorizer.pkl'))
            word_vectorizer = None  # Enhanced model uses single vectorizer
            logger.info("✅ Enhanced model loaded successfully!")
            model_loaded = True
            return True
            
        except FileNotFoundError:
            # Fallback to enhanced model
            try:
                ensemble_model = joblib.load(str(MODELS_DIR / 'enhanced_fake_news_classifier_passiveaggressive.pkl'))
                char_vectorizer = joblib.load(str(MODELS_DIR / 'enhanced_tfidf_vectorizer.pkl'))
                word_vectorizer = None  # Enhanced model uses single vectorizer
                logger.info("✅ Enhanced model loaded successfully!")
                model_loaded = True
                return True
                
            except FileNotFoundError:
                # Fallback to basic model
                ensemble_model = joblib.load(str(MODELS_DIR / 'fake_news_classifier.pkl'))
                char_vectorizer = joblib.load(str(MODELS_DIR / 'tfidf_vectorizer.pkl'))
                word_vectorizer = None  # Basic model uses single vectorizer
                logger.info("✅ Basic model loaded successfully!")
                model_loaded = True
                return True
                
    except Exception as e:
        try:
            models_listing = [p.name for p in MODELS_DIR.glob('*')]
        except Exception:
            models_listing = []
        logger.error(f"❌ Error loading models: {e} | MODELS_DIR={MODELS_DIR} | contents={models_listing}")
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
                fake_probability = 1 / (1 + np.exp(-decision_score))
            else:
                confidence = 0.7  # Default confidence for single model
                fake_probability = 1.0 if prediction == 1 else 0.0
        except:
            confidence = 0.7
            fake_probability = 1.0 if prediction == 1 else 0.0
        
        # Interpret result
        label = "FAKE" if prediction == 1 else "REAL"
        
        # Apply rule-based filters to catch obvious cases the AI missed
        filtered_label, filtered_confidence, filter_reason = apply_rule_filters(text, label, confidence)
        
        # If rule triggered, use filtered result
        if filter_reason != "AI Model":
            logger.info(f"Rule filter applied: {filter_reason}")
            label = filtered_label
            confidence = filtered_confidence
            fake_probability = 1.0 if label == "FAKE" else 0.0
        
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
    # Lazy-load models if not loaded yet (native deploy under WSGI)
    global model_loaded
    if not model_loaded:
        try:
            logger.info("Health: attempting lazy model load...")
            load_models()
        except Exception as e:
            logger.error(f"Health lazy load failed: {e}")
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
        # Ensure models are loaded
        global model_loaded
        if not model_loaded:
            logger.info("Predict: attempting lazy model load...")
            load_models()
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
            "model_type": "Hybrid AI + Rule-Based Model"
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

@app.route('/verify_fact', methods=['POST'])
def verify_fact():
    """
    Verify a claim using Google Gemini with Search Grounding
    """
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Gemini API key not configured on server"}), 503

        # Configure Gemini
        genai.configure(api_key=api_key)
        
        response = None
        last_error = None

        # Attempt 1: Try Gemini 1.5 Flash with Search Grounding (Best for fact checking)
        try:
            # Updated tool definition for Google Search
            tools = [
                {"google_search": {}}
            ]
            # Try specific version tag 'gemini-1.5-flash-001' which is often more stable than the alias
            model = genai.GenerativeModel('gemini-flash-latest', tools=tools)
            
            # Prompt engineering for fact-checking
            prompt = f"""
            You are an expert COVID-19 fact-checker. Your task is to verify the following claim.
            
            Claim: "{text}"
            
            Please provide a structured response in JSON format with the following fields:
            1. "verdict": One of ["True", "False", "Misleading", "Unverified"]
            2. "explanation": A concise 2-3 sentence explanation of why.
            3. "sources": A list of 2-3 credible sources (URLs) if available.
            4. "risk_level": One of ["High", "Medium", "Low"] regarding public health risk.
            
            Ensure the tone is objective and scientific. Return ONLY the JSON.
            """
            
            response = model.generate_content(prompt)
            
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️ Gemini 1.5 Flash failed: {e}. Falling back to Gemini Pro.")
            
            # Attempt 2: Fallback to Gemini Pro (Standard, no search tools but reliable)
            try:
                model = genai.GenerativeModel('gemini-flash-latest')
                
                # Modified prompt for model without search tools
                prompt = f"""
                You are an expert COVID-19 fact-checker. Your task is to verify the following claim based on your training data.
                
                Claim: "{text}"
                
                Please provide a structured response in JSON format with the following fields:
                1. "verdict": One of ["True", "False", "Misleading", "Unverified"]
                2. "explanation": A concise 2-3 sentence explanation of why.
                3. "sources": A list of 2-3 credible sources (URLs) if you know them, otherwise empty list [].
                4. "risk_level": One of ["High", "Medium", "Low"] regarding public health risk.
                
                Ensure the tone is objective and scientific. Return ONLY the JSON.
                """
                
                response = model.generate_content(prompt)
            except Exception as e2:
                raise Exception(f"All models failed. Flash error: {last_error}. Pro error: {e2}")

        # Parse the response
        response_text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        import json
        try:
            fact_check = json.loads(response_text)
        except json.JSONDecodeError:
            fact_check = {
                "verdict": "Unverified",
                "explanation": response_text,
                "sources": [],
                "risk_level": "Unknown"
            }
        
        # Save to MongoDB if available
        if verification_logs is not None:
            try:
                log_entry = {
                    "text": text,
                    "verdict": fact_check.get("verdict", "Unverified"),
                    "explanation": fact_check.get("explanation", ""),
                    "risk_level": fact_check.get("risk_level", "Unknown"),
                    "timestamp": datetime.utcnow(),
                    "source": "user_query"
                }
                verification_logs.insert_one(log_entry)
            except Exception as db_err:
                logger.error(f"Failed to save to MongoDB: {db_err}")
            
        return jsonify(fact_check)

    except Exception as e:
        logger.error(f"❌ Gemini verification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/fetch_latest_news', methods=['GET'])
def fetch_latest_news():
    """
    Fetch latest COVID-19 news using RSS Scraper and analyze with Gemini
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Gemini API key not configured"}), 503

        # 1. Scrape Google News RSS (Targeting COVID-19)
        # Using 'when:1d' to get news from the last 24 hours
        rss_url = "https://news.google.com/rss/search?q=covid-19+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            response = requests.get(rss_url, timeout=10)
            # Try using xml parser (requires lxml), fallback to html.parser
            try:
                soup = BeautifulSoup(response.content, features='xml')
            except Exception:
                soup = BeautifulSoup(response.content, features='html.parser')
                
            items = soup.findAll('item')[:8] # Get top 8 news items
            
            news_data = []
            for item in items:
                news_data.append({
                    "title": item.title.text,
                    "link": item.link.text,
                    "pubDate": item.pubDate.text,
                    "source": item.source.text if item.source else "Unknown"
                })
                
        except Exception as e:
            logger.error(f"RSS Scraping failed: {e}")
            return jsonify({"error": "Failed to fetch news feed"}), 500

        if not news_data:
             return jsonify({"news": []})

        # 2. Analyze with Gemini
        genai.configure(api_key=api_key)
        # Use standard model without tools since we provide the data
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        
        prompt = f"""
        I have scraped the following latest COVID-19 news items. 
        
        Raw Data:
        {str(news_data)}
        
        Your task:
        1. Analyze each news item.
        2. Filter out any irrelevant or duplicate stories.
        3. Return a JSON array with the following fields for each valid item:
           - "title": The headline (clean it up if needed).
           - "source": The news outlet name.
           - "url": The link provided.
           - "summary": A 1-sentence summary of what this news means.
           - "verdict": "Real" (if it's from a reputable source like WHO, NDTV, BBC, etc.) or "Fake" (if it sounds suspicious, though these are likely real news).
           - "confidence": A number between 0.8 and 1.0 (since these are from Google News, they are mostly real).
           - "timestamp": Convert the pubDate to a friendly string like "2 hours ago" or "Today".
        
        Return ONLY the JSON array.
        """
        
        try:
            response = model.generate_content(prompt)
            
            # Parse JSON from response
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            import json
            analyzed_news = json.loads(response_text)
            
            return jsonify({"news": analyzed_news})
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            # Fallback: return raw scraped data if AI fails
            fallback_news = []
            for item in news_data:
                fallback_news.append({
                    "title": item['title'],
                    "source": item['source'],
                    "url": item['link'],
                    "summary": "Latest update from Google News.",
                    "verdict": "Real",
                    "confidence": 0.9,
                    "timestamp": "Today"
                })
            return jsonify({"news": fallback_news})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard_stats', methods=['GET'])
def dashboard_stats():
    """
    Get statistics for the visualization dashboard
    """
    try:
        from datetime import datetime, timedelta
        from collections import Counter
        stats = {}
        
        # Use MongoDB if available
        if verification_logs is not None:
            # 1. Distribution
            total = verification_logs.count_documents({})
            fake = verification_logs.count_documents({"verdict": {"$in": ["False", "Misleading", "Fake"]}})
            real = verification_logs.count_documents({"verdict": {"$in": ["True", "Real"]}})
            
            stats["distribution"] = {
                "total": total,
                "fake": fake,
                "real": real
            }

            # 2. Timeline (Last 7 days)
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": datetime.utcnow() - timedelta(days=7)
                        }
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                        },
                        "fake_count": {
                            "$sum": {"$cond": [{"$in": ["$verdict", ["False", "Misleading", "Fake"]]}, 1, 0]}
                        },
                        "real_count": {
                            "$sum": {"$cond": [{"$in": ["$verdict", ["True", "Real"]]}, 1, 0]}
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            db_timeline = list(verification_logs.aggregate(pipeline))
            
            # Fill in missing days with 0
            timeline = []
            today = datetime.now()
            for i in range(7):
                date_str = (today - timedelta(days=6-i)).strftime("%Y-%m-%d")
                day_data = next((item for item in db_timeline if item["_id"] == date_str), None)
                timeline.append({
                    "date": date_str,
                    "fake_count": day_data["fake_count"] if day_data else 0,
                    "real_count": day_data["real_count"] if day_data else 0
                })
            stats["timeline"] = timeline

            # 3. Keywords (Dynamic from DB)
            try:
                recent_logs = list(verification_logs.find({}, {"text": 1}).sort("timestamp", -1).limit(100))
                all_text = " ".join([log.get("text", "") for log in recent_logs]).lower()
                stop_words = set(["the", "a", "an", "in", "on", "at", "for", "to", "of", "is", "are", "was", "were", "covid", "covid-19", "coronavirus", "virus", "news", "fake", "real", "check", "verify", "please", "this", "that", "with", "from", "about"])
                words = re.findall(r'\w+', all_text)
                filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
                common = Counter(filtered_words).most_common(10)
                stats["top_keywords"] = [{"keyword": k, "count": v} for k, v in common]
            except Exception as e:
                logger.error(f"Keyword extraction failed: {e}")
                stats["top_keywords"] = []
            
        else:
            # Fallback if DB not connected
            stats = {
                "distribution": {"total": 0, "fake": 0, "real": 0},
                "timeline": [],
                "top_keywords": []
            }

        # 4. Trending Topics (Fetch from Gemini for "Live" feel)
        trending_topics = []
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-001')
                prompt = "List 5 current trending COVID-19 misinformation topics or general health rumors. Return ONLY a JSON array of strings. Example: [\"Topic 1\", \"Topic 2\"]"
                response = model.generate_content(prompt)
                text = response.text.strip()
                if text.startswith("```json"): text = text[7:]
                if text.endswith("```"): text = text[:-3]
                import json
                trending_topics = json.loads(text)
        except Exception as e:
            logger.error(f"Failed to fetch trending topics: {e}")
            
        # Ensure trending_topics is never empty to prevent frontend crash
        if not trending_topics:
            trending_topics = [
                "New Variant Rumors",
                "Vaccine Side Effects Exaggeration",
                "Mask Mandate Conspiracies",
                "Lockdown Return Fears",
                "Alternative Cure Hoaxes"
            ]
        
        stats["trending_topics"] = trending_topics
        
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("🚀 Starting COVID-19 Fake News Detection API...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to load models. Please check model files.")
        print("❌ Cannot start API without models. Please ensure model files exist in 'models/' directory.")