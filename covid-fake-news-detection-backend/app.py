from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
import os
import io
from datetime import datetime
import logging
from scipy.sparse import hstack
from pathlib import Path
from urllib.parse import urlparse
import json
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
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173"
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
easyocr_reader = None

DEFAULT_MODALITY_WEIGHTS = {
    "text": 0.5,
    "image": 0.3,
    "metadata": 0.2
}

# Model fallback order requested for quota resilience.
GEMINI_MODEL_FALLBACKS = [
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
]

AVAILABLE_GEMINI_MODELS_CACHE = None

def _normalize_model_name(model_name):
    if not model_name:
        return ""
    return model_name.replace("models/", "").strip()

def _get_available_generate_content_models():
    """
    Fetch and cache models that support generateContent for this API version.
    """
    global AVAILABLE_GEMINI_MODELS_CACHE
    if AVAILABLE_GEMINI_MODELS_CACHE is not None:
        return AVAILABLE_GEMINI_MODELS_CACHE

    try:
        discovered = set()
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                discovered.add(_normalize_model_name(getattr(model, "name", "")))
        AVAILABLE_GEMINI_MODELS_CACHE = discovered
        logger.info(f"Discovered {len(discovered)} Gemini models supporting generateContent.")
        return discovered
    except Exception as e:
        logger.warning(f"Could not list Gemini models; using static fallback list. Error: {e}")
        AVAILABLE_GEMINI_MODELS_CACHE = set()
        return AVAILABLE_GEMINI_MODELS_CACHE

def generate_with_gemini_fallback(prompt, tools=None):
    """
    Generate content using Gemini with automatic model fallback:
    gemini-3-flash -> gemini-2.5-flash -> gemini-2.0-flash.
    """
    models_to_try = GEMINI_MODEL_FALLBACKS.copy()
    last_error = None

    # Optional override via env, while preserving fallback defaults.
    preferred_model = os.getenv("GEMINI_PRIMARY_MODEL", "").strip()
    if preferred_model and preferred_model not in models_to_try:
        models_to_try.insert(0, preferred_model)

    available_models = _get_available_generate_content_models()
    if available_models:
        filtered = [m for m in models_to_try if _normalize_model_name(m) in available_models]
        if filtered:
            models_to_try = filtered
        else:
            logger.warning("No requested fallback models were found in available API models; trying static list anyway.")

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name, tools=tools) if tools else genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response, model_name
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini model {model_name} failed, trying next fallback. Error: {e}")

            # If tools were enabled and failed, try same model once without tools.
            if tools:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    logger.warning(f"Gemini model {model_name} succeeded without tools after tool-mode failure.")
                    return response, f"{model_name} (no-tools)"
                except Exception as e_no_tools:
                    last_error = e_no_tools
                    logger.warning(f"Gemini model {model_name} without tools also failed: {e_no_tools}")

    raise Exception(f"All Gemini fallback models failed. Last error: {last_error}")

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

def normalize_confidence(raw_confidence):
    """
    Normalize model confidence to 0-1 range.
    """
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.5

    if confidence > 1.0:
        # decision_function confidence can exceed 1
        confidence = 1 / (1 + np.exp(-confidence))
    return max(0.0, min(1.0, confidence))

def score_text_modality(text):
    """
    Score text modality using existing ML pipeline.
    """
    if not text or len(text.strip()) < 10:
        return {
            "available": False,
            "fake_risk": 0.5,
            "confidence": 0.0,
            "notes": ["Not enough text supplied for text model"]
        }

    label, confidence, _, fake_probability = predict_news(text)
    if label is None:
        return {
            "available": False,
            "fake_risk": 0.5,
            "confidence": 0.0,
            "notes": ["Text model failed to generate a prediction"]
        }

    return {
        "available": True,
        "prediction": label,
        "fake_risk": float(fake_probability),
        "confidence": normalize_confidence(confidence),
        "notes": [f"Text model predicted {label}"]
    }

def extract_text_from_image_bytes(image_bytes):
    """
    Extract text from image using OCR, if OCR dependencies are available.
    """
    global easyocr_reader

    if not image_bytes:
        return "", "No image bytes provided"

    try:
        from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    except ImportError:
        return "", "Pillow not installed (OCR unavailable)"

    try:
        import easyocr
    except ImportError:
        return "", "easyocr not installed (OCR unavailable)"

    try:
        if easyocr_reader is None:
            # Keep OCR CPU-friendly for deployment environments.
            easyocr_reader = easyocr.Reader(["en"], gpu=False)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Upscale small images to improve OCR quality for screenshots/posts.
        if image.width < 1200:
            scale = max(1.5, 1200 / max(1, image.width))
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS
            )

        gray = ImageOps.grayscale(image)
        high_contrast = ImageOps.autocontrast(gray)
        sharpened = high_contrast.filter(ImageFilter.SHARPEN)
        boosted = ImageEnhance.Contrast(sharpened).enhance(1.8)
        thresholded = boosted.point(lambda p: 255 if p > 150 else 0)

        candidate_images = [image, gray, boosted, thresholded]

        best_text = ""
        best_score = 0

        for candidate in candidate_images:
            try:
                result = easyocr_reader.readtext(np.array(candidate), detail=0, paragraph=True)
                raw = " ".join(result) if result else ""
            except Exception:
                continue
            cleaned = re.sub(r"\s+", " ", raw).strip()
            # Score by alpha-numeric signal (more likely meaningful OCR text).
            score = len(re.findall(r"[A-Za-z0-9]", cleaned))
            if score > best_score:
                best_score = score
                best_text = cleaned

        if best_text and best_score >= 12:
            return best_text, "EasyOCR extracted text from enhanced image processing"
        return "", "OCR found no readable text. Try a clearer image with larger visible text."
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return "", f"OCR failed: {str(e)}"

def compute_visual_risk_from_image(image_bytes):
    """
    Lightweight visual signal from image structure (contrast and edge density).
    """
    if not image_bytes:
        return 0.5, "No image provided for visual analysis"

    try:
        from PIL import Image
    except ImportError:
        return 0.5, "Pillow not installed (visual analysis unavailable)"

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        gray = np.asarray(image, dtype=np.float32)
        if gray.size == 0:
            return 0.5, "Image decoding failed"

        contrast_score = min(1.0, float(np.std(gray) / 64.0))
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edge_density = float((np.mean(gx > 30) + np.mean(gy > 30)) / 2.0)
        edge_score = min(1.0, edge_density * 3.0)

        visual_risk = float((0.45 * contrast_score) + (0.55 * edge_score))
        visual_risk = max(0.0, min(1.0, visual_risk))
        return visual_risk, "Visual score from edge density and contrast"
    except Exception as e:
        logger.warning(f"Visual analysis failed: {e}")
        return 0.5, f"Visual analysis failed: {str(e)}"

def score_image_modality(image_bytes):
    """
    Score image modality using OCR + lightweight visual signal.
    """
    if not image_bytes:
        return {
            "available": False,
            "fake_risk": 0.5,
            "confidence": 0.0,
            "ocr_text": "",
            "notes": ["No image provided"]
        }

    ocr_text, ocr_note = extract_text_from_image_bytes(image_bytes)
    visual_risk, visual_note = compute_visual_risk_from_image(image_bytes)
    notes = [ocr_note, visual_note]

    ocr_risk = None
    ocr_confidence = 0.0
    if ocr_text and len(ocr_text) >= 10:
        text_score = score_text_modality(ocr_text)
        if text_score["available"]:
            ocr_risk = text_score["fake_risk"]
            ocr_confidence = text_score["confidence"]
            notes.extend(text_score.get("notes", []))

    if ocr_risk is not None:
        # Weight OCR higher because it captures semantic claim text from image.
        fake_risk = (0.7 * ocr_risk) + (0.3 * visual_risk)
        confidence = (0.7 * ocr_confidence) + 0.2
    else:
        fake_risk = visual_risk
        confidence = 0.4
        notes.append("Image OCR text unavailable; using visual-only signal")

    return {
        "available": True,
        "fake_risk": float(max(0.0, min(1.0, fake_risk))),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "ocr_text": ocr_text,
        "notes": notes
    }

def score_metadata_modality(source_url, text):
    """
    Score metadata modality from URL/domain and language cues.
    """
    signals = []
    risk = 0.35
    available = False

    if source_url:
        available = True
        try:
            parsed = urlparse(source_url.strip())
            domain = (parsed.netloc or "").lower()
            path = (parsed.path or "").lower()
            full = f"{domain}{path}"

            if parsed.scheme != "https":
                risk += 0.15
                signals.append("Source URL is not HTTPS")

            suspicious_terms = ["breaking", "shocking", "exclusive", "miracle", "cure", "truth", "secret"]
            if any(term in full for term in suspicious_terms):
                risk += 0.15
                signals.append("URL contains clickbait-like keywords")

            if domain.count(".") >= 3:
                risk += 0.1
                signals.append("URL has deep subdomain nesting")

            if domain.count("-") >= 2 or sum(ch.isdigit() for ch in domain) >= 3:
                risk += 0.1
                signals.append("Domain contains unusual characters/digits")
        except Exception as e:
            logger.warning(f"Metadata URL parse failed: {e}")
            signals.append("URL parsing failed")

    if text and text.strip():
        available = True
        text_lower = text.lower()
        if text.count("!") >= 3:
            risk += 0.08
            signals.append("Text uses excessive exclamation marks")

        sensational_phrases = ["they don't want you to know", "hidden truth", "100% cure", "guaranteed cure"]
        if any(phrase in text_lower for phrase in sensational_phrases):
            risk += 0.15
            signals.append("Text has sensational claim patterns")

    if not available:
        return {
            "available": False,
            "fake_risk": 0.5,
            "confidence": 0.0,
            "notes": ["No URL/text metadata supplied"]
        }

    if not signals:
        signals.append("No strong metadata risk signals detected")

    return {
        "available": True,
        "fake_risk": float(max(0.0, min(1.0, risk))),
        "confidence": 0.55,
        "notes": signals
    }

def fuse_modalities(modality_scores):
    """
    Weighted late-fusion across available modalities.
    """
    available_modalities = {
        name: values for name, values in modality_scores.items()
        if values.get("available")
    }

    if not available_modalities:
        return 0.5, {}

    active_weight_sum = sum(DEFAULT_MODALITY_WEIGHTS[name] for name in available_modalities)
    normalized_weights = {
        name: DEFAULT_MODALITY_WEIGHTS[name] / active_weight_sum
        for name in available_modalities
    }

    fused_risk = 0.0
    for name, values in available_modalities.items():
        fused_risk += normalized_weights[name] * values["fake_risk"]

    return float(max(0.0, min(1.0, fused_risk))), normalized_weights

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

@app.route('/predict_multimodal', methods=['POST'])
def predict_multimodal():
    """
    Multimodal prediction endpoint: text + image + metadata URL.
    """
    try:
        global model_loaded
        if not model_loaded:
            logger.info("Predict multimodal: attempting lazy model load...")
            load_models()

        text = (request.form.get('text') or "").strip()
        source_url = (request.form.get('url') or "").strip()
        image_file = request.files.get('image')

        has_image = bool(image_file and image_file.filename)
        if not text and not source_url and not has_image:
            return jsonify({
                "error": "Provide at least one input: text, image, or url."
            }), 400

        image_bytes = b""
        if has_image:
            image_bytes = image_file.read()
            if not image_bytes:
                return jsonify({"error": "Uploaded image is empty."}), 400

        text_score = score_text_modality(text)
        image_score = score_image_modality(image_bytes)
        metadata_score = score_metadata_modality(source_url, text)

        modality_scores = {
            "text": text_score,
            "image": image_score,
            "metadata": metadata_score
        }
        fused_fake_risk, normalized_weights = fuse_modalities(modality_scores)

        image_ocr_text = image_score.get("ocr_text", "")
        image_is_covid_related = None
        image_relevance_reason = "No image provided."
        if has_image:
            if image_ocr_text and len(image_ocr_text.strip()) >= 10:
                image_is_covid_related = quick_covid_relevance_check(image_ocr_text)
                if image_is_covid_related:
                    image_relevance_reason = "Image OCR text appears related to COVID-19."
                else:
                    image_relevance_reason = "Image OCR text is not related to COVID-19."
            else:
                image_relevance_reason = "No readable text found in image OCR. Please upload a clearer image."

        prediction = "FAKE" if fused_fake_risk >= 0.5 else "REAL"
        confidence = min(1.0, 0.5 + abs(fused_fake_risk - 0.5))

        if confidence >= 0.85:
            confidence_level = "Very High"
        elif confidence >= 0.7:
            confidence_level = "High"
        elif confidence >= 0.55:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        response = {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "confidence_level": confidence_level,
            "fake_probability": round(fused_fake_risk * 100, 1),
            "real_probability": round((1 - fused_fake_risk) * 100, 1),
            "timestamp": datetime.now().isoformat(),
            "model_type": "Multimodal Fusion (Text + OCR/Image + Metadata)",
            "modality_scores": {
                name: {
                    "available": values.get("available", False),
                    "fake_risk": round(values.get("fake_risk", 0.5), 3),
                    "confidence": round(values.get("confidence", 0.0), 3),
                    "weight": round(normalized_weights.get(name, 0.0), 3),
                    "notes": values.get("notes", [])[:3]
                }
                for name, values in modality_scores.items()
            },
            "ocr_text_preview": (
                image_score.get("ocr_text", "")[:300] + "..."
                if len(image_score.get("ocr_text", "")) > 300
                else image_score.get("ocr_text", "")
            ),
            "ocr_text_for_verification": image_ocr_text[:6000] if image_ocr_text else "",
            "image_has_text": bool(image_ocr_text.strip()),
            "image_is_covid_related": image_is_covid_related,
            "image_relevance_reason": image_relevance_reason
        }

        logger.info(
            f"Multimodal prediction: {prediction} | risk={fused_fake_risk:.3f} | "
            f"active_weights={normalized_weights} | "
            f"image_has_text={response['image_has_text']} | "
            f"ocr_text_length={len(image_ocr_text)} | "
            f"image_is_covid_related={response['image_is_covid_related']}"
        )
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Multimodal API error: {e}")
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

def extract_article_from_url(url):
    """
    Fetch and extract basic article content from a URL.
    """
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"ok": False, "error": "Invalid URL format"}

        response = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove non-content tags.
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        text_chunks = [chunk.strip() for chunk in soup.stripped_strings if chunk and len(chunk.strip()) > 2]
        article_text = " ".join(text_chunks)
        article_text = re.sub(r"\s+", " ", article_text).strip()

        if not article_text:
            return {"ok": False, "error": "Could not extract article text from URL"}

        return {
            "ok": True,
            "title": title,
            "content": article_text[:15000]  # Keep payload bounded for model call
        }
    except Exception as e:
        logger.warning(f"URL scraping failed for {url}: {e}")
        return {"ok": False, "error": f"URL scraping failed: {str(e)}"}

def quick_covid_relevance_check(text_blob):
    """
    Fast keyword-based COVID relevance check for fallback and pre-filter.
    """
    if not text_blob:
        return False

    text = text_blob.lower()
    covid_terms = [
        "covid", "covid-19", "coronavirus", "sars-cov-2", "pandemic",
        "lockdown", "mask mandate", "vaccination", "vaccine", "booster",
        "who", "cdc", "variant", "omicron", "quarantine"
    ]
    return any(term in text for term in covid_terms)

def parse_gemini_json_response(response_text, default_payload):
    """
    Parse Gemini response robustly even with markdown fencing.
    """
    if not response_text:
        return default_payload

    payload = response_text.strip()
    if payload.startswith("```json"):
        payload = payload[7:]
    if payload.endswith("```"):
        payload = payload[:-3]
    payload = payload.strip()

    try:
        return json.loads(payload)
    except Exception:
        return default_payload

@app.route('/verify_fact', methods=['POST'])
def verify_fact():
    """
    Verify text and/or URL content using Gemini.
    If URL is provided, scrape it and validate COVID relevance first.
    """
    try:
        data = request.get_json() or {}
        text = (data.get('text') or "").strip()
        source_url = (data.get('url') or "").strip()
        extracted_title = ""
        extracted_content = ""
        scraped_ok = False
        scrape_error = None

        if not text and not source_url:
            return jsonify({"error": "No text or url provided"}), 400

        # If URL is provided, scrape and extract article content first.
        if source_url:
            scrape_result = extract_article_from_url(source_url)
            scraped_ok = scrape_result.get("ok", False)
            if scraped_ok:
                extracted_title = scrape_result.get("title", "")
                extracted_content = scrape_result.get("content", "")
            else:
                scrape_error = scrape_result.get("error", "Unable to scrape URL")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("Gemini API key not configured; returning graceful fallback.")
            return jsonify({
                "verdict": "Unverified",
                "explanation": "Gemini API key is not configured on the server. Showing multimodal model output only.",
                "sources": [],
                "risk_level": "Unknown",
                "service_status": "degraded",
                "is_covid_related": None,
                "url_checked": bool(source_url),
                "error": "Gemini API key not configured on server"
            }), 200

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Determine verification content
        content_for_check = text
        if scraped_ok and extracted_content:
            content_for_check = f"Title: {extracted_title}\n\nArticle Content:\n{extracted_content}"

        if not content_for_check:
            return jsonify({
                "verdict": "Unverified",
                "explanation": scrape_error or "No verifiable content found.",
                "sources": [],
                "risk_level": "Unknown",
                "is_covid_related": None,
                "url_checked": bool(source_url)
            }), 200

        # COVID relevance gate when URL is used.
        covid_related = quick_covid_relevance_check(content_for_check)
        relevance_reason = "Detected COVID-related terms in content."

        if source_url:
            try:
                relevance_prompt = f"""
                You are checking whether an article is related to COVID-19 / coronavirus public-health topic.
                Return ONLY JSON with:
                {{
                  "is_covid_related": true/false,
                  "reason": "short reason"
                }}

                Article URL: {source_url}
                Title: {extracted_title}
                Article excerpt:
                {extracted_content[:3500]}
                """
                relevance_response, relevance_model_used = generate_with_gemini_fallback(relevance_prompt)
                relevance_json = parse_gemini_json_response(
                    getattr(relevance_response, "text", ""),
                    {
                        "is_covid_related": covid_related,
                        "reason": relevance_reason
                    }
                )
                covid_related = bool(relevance_json.get("is_covid_related", covid_related))
                relevance_reason = relevance_json.get("reason", relevance_reason)
                logger.info(f"Gemini relevance model used: {relevance_model_used}")
            except Exception as rel_err:
                logger.warning(f"Gemini relevance check failed, using keyword fallback: {rel_err}")

            if not covid_related:
                non_covid_response = {
                    "verdict": "Not Related",
                    "explanation": "The provided URL does not appear to be related to COVID-19. Please enter a COVID-related article URL for verification.",
                    "sources": [],
                    "risk_level": "Low",
                    "is_covid_related": False,
                    "url_checked": True,
                    "url": source_url,
                    "relevance_reason": relevance_reason
                }
                return jsonify(non_covid_response), 200

        # Main Gemini verification.
        # Search grounding can be toggled by env to avoid SDK/tool incompatibility noise.
        response = None
        model_used = None
        try:
            enable_search_tools = os.getenv("ENABLE_GEMINI_SEARCH_TOOLS", "false").lower() == "true"
            tools = [{"google_search": {}}] if enable_search_tools else None
            prompt = f"""
            You are an expert COVID-19 fact-checker. Verify the following COVID-related claim/content.

            Content to verify:
            {content_for_check[:6000]}

            Return ONLY JSON with fields:
            1. "verdict": one of ["True", "False", "Misleading", "Unverified"]
            2. "explanation": concise 2-3 sentence explanation.
            3. "sources": list of 2-4 credible source URLs.
            4. "risk_level": one of ["High", "Medium", "Low"]
            """
            response, model_used = generate_with_gemini_fallback(prompt, tools=tools)
        except Exception as e:
            raise Exception(f"Gemini verification failed across fallback models: {e}")

        fact_check = parse_gemini_json_response(
            getattr(response, "text", ""),
            {
                "verdict": "Unverified",
                "explanation": "Could not parse Gemini response.",
                "sources": [],
                "risk_level": "Unknown"
            }
        )
        fact_check["is_covid_related"] = True
        fact_check["url_checked"] = bool(source_url)
        fact_check["gemini_model_used"] = model_used
        if source_url:
            fact_check["url"] = source_url
            fact_check["url_scraped"] = scraped_ok
            if scrape_error:
                fact_check["url_scrape_note"] = scrape_error
            if extracted_title:
                fact_check["article_title"] = extracted_title
        
        # Save to MongoDB if available
        if verification_logs is not None:
            try:
                log_entry = {
                    "text": text or extracted_title or source_url,
                    "verdict": fact_check.get("verdict", "Unverified"),
                    "explanation": fact_check.get("explanation", ""),
                    "risk_level": fact_check.get("risk_level", "Unknown"),
                    "timestamp": datetime.utcnow(),
                    "source": "user_query",
                    "url": source_url if source_url else None,
                    "is_covid_related": fact_check.get("is_covid_related")
                }
                verification_logs.insert_one(log_entry)
            except Exception as db_err:
                logger.error(f"Failed to save to MongoDB: {db_err}")
            
        return jsonify(fact_check)

    except Exception as e:
        logger.error(f"❌ Gemini verification error: {e}")
        fallback = {
            "verdict": "Unverified",
            "explanation": "verification service is temporarily unavailable. Falling back to multimodal model output.",
            "sources": [],
            "risk_level": "Unknown",
            "service_status": "degraded",
            "error": str(e)
        }
        return jsonify(fallback), 200

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
            response, model_used = generate_with_gemini_fallback(prompt)
            
            # Parse JSON from response
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            import json
            analyzed_news = json.loads(response_text)
            
            return jsonify({"news": analyzed_news, "gemini_model_used": model_used})
            
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
                prompt = "List 5 current trending COVID-19 misinformation topics or general health rumors. Return ONLY a JSON array of strings. Example: [\"Topic 1\", \"Topic 2\"]"
                response, _model_used = generate_with_gemini_fallback(prompt)
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

# Load models when the module is imported (for Gunicorn)
load_models()

if __name__ == '__main__':
    # For local development
    logger.info("🚀 Starting COVID-19 Fake News Detection API...")
    app.run(debug=True, host='0.0.0.0', port=5000)