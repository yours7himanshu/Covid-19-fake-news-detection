import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    # Remove quotes, newlines, and extra whitespace
    text = str(text).replace('"', '').replace('\n', ' ').strip()
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_prepare_data():
    """
    Load fake and real COVID-19 datasets and prepare for training
    """
    try:
        # Load fake claims dataset
        fake_df = pd.read_csv("./datasets/ClaimFakeCOVID-19_5.csv")
        print(f"Loaded fake claims dataset: {len(fake_df)} samples")
        
        # Load real claims dataset  
        real_df = pd.read_csv("./datasets/ClaimRealCOVID-19_5.csv")
        print(f"Loaded real claims dataset: {len(real_df)} samples")
        
        # Select relevant columns and add labels
        fake_df = fake_df[['title']].copy()
        fake_df['label'] = 1  # 1 for fake
        fake_df['category'] = 'fake'
        
        real_df = real_df[['title']].copy()  
        real_df['label'] = 0  # 0 for real
        real_df['category'] = 'real'
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Remove missing values
        df.dropna(inplace=True)
        
        # Preprocess text
        df['title_clean'] = df['title'].apply(preprocess_text)
        
        # Remove empty titles after preprocessing
        df = df[df['title_clean'].str.len() > 0]
        
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Fake news samples: {len(df[df['label'] == 1])}")
        print(f"Real news samples: {len(df[df['label'] == 0])}")
        print(f"\nFirst 5 samples:")
        print(df[['title_clean', 'label', 'category']].head())
        
        return df
        
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        return None

def train_model(df):
    """
    Train the fake news detection model
    """
    # Prepare features and labels
    X = df['title_clean']
    y = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF feature shape: {tfidf_train.shape}")
    
    # Initialize and train the PassiveAggressive Classifier
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    pac.fit(tfidf_train, y_train)
    
    # Make predictions
    y_pred = pac.predict(tfidf_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return pac, tfidf_vectorizer, accuracy

def save_model(model, vectorizer):
    """
    Save the trained model and vectorizer
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model and vectorizer
        joblib.dump(model, 'models/fake_news_classifier.pkl')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        
        print(f"\nModel saved successfully!")
        print(f"- Classifier: models/fake_news_classifier.pkl")
        print(f"- Vectorizer: models/tfidf_vectorizer.pkl")
        
    except Exception as e:
        print(f"Error saving model: {e}")

def test_model_prediction(model, vectorizer):
    """
    Test the model with some sample predictions
    """
    print(f"\n" + "="*50)
    print("TESTING MODEL WITH SAMPLE PREDICTIONS")
    print("="*50)
    
    # Test samples
    test_samples = [
        "COVID-19 vaccines alter your DNA permanently",  # Should be fake
        "Wearing masks can help reduce the spread of COVID-19",  # Should be real
        "5G towers cause coronavirus infection",  # Should be fake
        "Washing hands frequently helps prevent COVID-19",  # Should be real
        "Drinking bleach cures coronavirus"  # Should be fake
    ]
    
    for i, sample in enumerate(test_samples, 1):
        # Preprocess the sample
        processed_sample = preprocess_text(sample)
        
        # Vectorize the sample
        sample_tfidf = vectorizer.transform([processed_sample])
        
        # Make prediction
        prediction = model.predict(sample_tfidf)[0]
        probability = model.decision_function(sample_tfidf)[0]
        
        # Interpret results
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = abs(probability)
        
        print(f"\nTest {i}:")
        print(f"Text: {sample}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.3f}")

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("COVID-19 FAKE NEWS DETECTION MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Train the model
    model, vectorizer, accuracy = train_model(df)
    
    # Save the model
    save_model(model, vectorizer)
    
    # Test with sample predictions
    test_model_prediction(model, vectorizer)
    
    print(f"\n" + "="*60)
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()