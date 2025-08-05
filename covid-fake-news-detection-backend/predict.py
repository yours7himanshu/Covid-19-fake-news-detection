import joblib
import re
import numpy as np

def preprocess_text(text):
    """
    Clean and preprocess text data (same as training)
    """
    text = str(text).replace('"', '').replace('\n', ' ').strip()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_model():
    """
    Load the trained model and vectorizer
    """
    try:
        # Try to load enhanced model first
        try:
            model = joblib.load('models/enhanced_fake_news_classifier_passiveaggressive.pkl')
            vectorizer = joblib.load('models/enhanced_tfidf_vectorizer.pkl')
            print("✅ Loaded enhanced model")
            return model, vectorizer
        except FileNotFoundError:
            # Fallback to basic model
            model = joblib.load('models/fake_news_classifier.pkl')
            vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            print("✅ Loaded basic model")
            return model, vectorizer
            
    except FileNotFoundError:
        print("❌ Error: No trained model found!")
        print("Please run 'python train_enhanced.py' first to train a model.")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_news(text, model, vectorizer):
    """
    Predict if a news text is fake or real
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    if len(processed_text.strip()) == 0:
        return None, 0.0, "Empty text after preprocessing"
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get confidence score
    try:
        if hasattr(model, 'decision_function'):
            confidence_raw = model.decision_function(text_vectorized)[0]
            confidence = abs(confidence_raw)
        elif hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0
    except:
        confidence = 1.0
    
    # Interpret result
    label = "FAKE" if prediction == 1 else "REAL"
    
    return label, confidence, processed_text

def interactive_mode():
    """
    Interactive mode for testing multiple texts
    """
    print("\n" + "="*60)
    print("🔍 COVID-19 FAKE NEWS DETECTOR - INTERACTIVE MODE")
    print("="*60)
    print("Enter news text to check if it's fake or real.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    while True:
        try:
            # Get user input
            text = input("\n📝 Enter news text: ").strip()
            
            # Check for exit commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not text:
                print("⚠️  Please enter some text")
                continue
            
            # Make prediction
            result, confidence, processed = predict_news(text, model, vectorizer)
            
            if result is None:
                print(f"❌ Error: {confidence}")
                continue
            
            # Display result with emoji and color coding
            if result == "FAKE":
                emoji = "🚨"
                status = "FAKE NEWS"
            else:
                emoji = "✅"
                status = "REAL NEWS"
            
            print(f"\n{emoji} Prediction: {status}")
            print(f"🎯 Confidence: {confidence:.3f}")
            print(f"📄 Processed text: {processed[:100]}{'...' if len(processed) > 100 else ''}")
            
            # Add interpretation
            if confidence > 1.5:
                certainty = "Very High"
            elif confidence > 1.0:
                certainty = "High"
            elif confidence > 0.5:
                certainty = "Medium"
            else:
                certainty = "Low"
            
            print(f"📊 Certainty: {certainty}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test():
    """
    Test with predefined examples
    """
    print("\n" + "="*60)
    print("🧪 COVID-19 FAKE NEWS DETECTOR - BATCH TEST")
    print("="*60)
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    # Test cases
    test_cases = [
        # Clearly fake
        ("COVID-19 vaccines alter your DNA permanently", "FAKE"),
        ("5G towers cause coronavirus infections", "FAKE"),
        ("Drinking bleach cures coronavirus", "FAKE"),
        ("Coronavirus is a hoax by the government", "FAKE"),
        ("Hydroxychloroquine is 100% effective against COVID-19", "FAKE"),
        
        # Clearly real
        ("Washing hands helps prevent COVID-19", "REAL"),
        ("Wearing masks can reduce coronavirus transmission", "REAL"),
        ("COVID-19 vaccines are safe and effective", "REAL"),
        ("Social distancing slows the spread of coronavirus", "REAL"),
        ("COVID-19 symptoms include fever and cough", "REAL"),
        
        # Nuanced cases
        ("Some people have mild reactions to COVID-19 vaccines", "REAL"),
        ("Young people are less likely to have severe COVID-19", "REAL"),
        ("Natural immunity provides some protection", "REAL"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print(f"Testing {total} cases...\n")
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result, confidence, _ = predict_news(text, model, vectorizer)
        
        if result is None:
            continue
            
        is_correct = result == expected
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} Test {i:2d}: {result:4s} (expected {expected:4s}) | Confidence: {confidence:.3f}")
        print(f"         Text: {text}")
        print()
    
    accuracy = correct / total if total > 0 else 0
    print(f"📊 Batch Test Accuracy: {correct}/{total} = {accuracy:.1%}")

def main():
    """
    Main function with menu
    """
    print("="*60)
    print("🔍 COVID-19 FAKE NEWS DETECTION SYSTEM")
    print("="*60)
    print("Choose an option:")
    print("1. Interactive mode (enter your own text)")
    print("2. Batch test (test with predefined examples)")
    print("3. Quick test (single prediction)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\n👉 Enter your choice (1-4): ").strip()
            
            if choice == '1':
                interactive_mode()
            elif choice == '2':
                batch_test()
            elif choice == '3':
                # Quick single test
                model, vectorizer = load_model()
                if model is None:
                    continue
                    
                text = input("\n📝 Enter news text: ").strip()
                if text:
                    result, confidence, processed = predict_news(text, model, vectorizer)
                    if result:
                        emoji = "🚨" if result == "FAKE" else "✅"
                        print(f"\n{emoji} Prediction: {result}")
                        print(f"🎯 Confidence: {confidence:.3f}")
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()