import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import os
import glob

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

def load_multiple_datasets():
    """
    Load multiple COVID-19 datasets and prepare for training
    """
    datasets = []
    
    # Load all available fake datasets
    fake_files = [
        "./datasets/ClaimFakeCOVID-19_5.csv",
        "./datasets/ClaimFakeCOVID-19_7.csv",
        "./datasets/NewsFakeCOVID-19_5.csv",
        "./datasets/NewsFakeCOVID-19_7.csv"
    ]
    
    # Load all available real datasets  
    real_files = [
        "./datasets/ClaimRealCOVID-19_5.csv",
        "./datasets/ClaimRealCOVID-19_7.csv", 
        "./datasets/NewsRealCOVID-19_5.csv",
        "./datasets/NewsRealCOVID-19_7.csv"
    ]
    
    print("Loading datasets...")
    
    # Load fake datasets
    fake_data = []
    for file in fake_files:
        try:
            df = pd.read_csv(file)
            if 'title' in df.columns:
                df_subset = df[['title']].copy()
                df_subset['label'] = 1  # 1 for fake
                df_subset['category'] = 'fake'
                df_subset['source'] = file.split('/')[-1]
                fake_data.append(df_subset)
                print(f"✓ Loaded {file}: {len(df)} samples")
            else:
                print(f"✗ Skipped {file}: No 'title' column")
        except FileNotFoundError:
            print(f"✗ File not found: {file}")
        except Exception as e:
            print(f"✗ Error loading {file}: {e}")
    
    # Load real datasets
    real_data = []
    for file in real_files:
        try:
            df = pd.read_csv(file)
            if 'title' in df.columns:
                df_subset = df[['title']].copy()
                df_subset['label'] = 0  # 0 for real
                df_subset['category'] = 'real'
                df_subset['source'] = file.split('/')[-1]
                real_data.append(df_subset)
                print(f"✓ Loaded {file}: {len(df)} samples")
            else:
                print(f"✗ Skipped {file}: No 'title' column")
        except FileNotFoundError:
            print(f"✗ File not found: {file}")
        except Exception as e:
            print(f"✗ Error loading {file}: {e}")
    
    # Combine all datasets
    all_data = []
    if fake_data:
        all_data.extend(fake_data)
    if real_data:
        all_data.extend(real_data)
    
    if not all_data:
        print("No datasets loaded successfully!")
        return None
        
    # Concatenate all dataframes
    df = pd.concat(all_data, ignore_index=True)
    
    # Remove missing values
    df.dropna(inplace=True)
    
    # Preprocess text
    df['title_clean'] = df['title'].apply(preprocess_text)
    
    # Remove empty titles after preprocessing
    df = df[df['title_clean'].str.len() > 10]  # Minimum 10 characters
    
    print(f"\n📊 Combined Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Fake news samples: {len(df[df['label'] == 1])}")
    print(f"Real news samples: {len(df[df['label'] == 0])}")
    
    # Show distribution by source
    print(f"\nDataset distribution by source:")
    source_dist = df.groupby(['source', 'category']).size().reset_index(name='count')
    for _, row in source_dist.iterrows():
        print(f"  {row['source']} ({row['category']}): {row['count']} samples")
    
    return df

def balance_dataset(df, method='smote'):
    """
    Balance the dataset to handle class imbalance
    """
    X = df['title_clean']
    y = df['label']
    
    if method == 'smote':
        # Use SMOTE for oversampling
        print(f"\n🔄 Applying SMOTE to balance the dataset...")
        
        # First vectorize the text
        temp_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_temp = temp_vectorizer.fit_transform(X)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(df[df['label']==1])-1))
        X_balanced, y_balanced = smote.fit_resample(X_temp, y)
        
        print(f"After SMOTE:")
        print(f"  Fake samples: {sum(y_balanced == 1)}")
        print(f"  Real samples: {sum(y_balanced == 0)}")
        
        return X_balanced, y_balanced, temp_vectorizer
    
    elif method == 'undersample':
        # Simple undersampling
        print(f"\n🔄 Applying undersampling to balance the dataset...")
        
        fake_samples = df[df['label'] == 1]
        real_samples = df[df['label'] == 0]
        
        # Sample equal number from majority class
        min_samples = min(len(fake_samples), len(real_samples))
        
        if len(fake_samples) < len(real_samples):
            real_samples = real_samples.sample(n=min_samples, random_state=42)
        else:
            fake_samples = fake_samples.sample(n=min_samples, random_state=42)
        
        balanced_df = pd.concat([fake_samples, real_samples], ignore_index=True)
        
        print(f"After undersampling:")
        print(f"  Fake samples: {len(balanced_df[balanced_df['label'] == 1])}")
        print(f"  Real samples: {len(balanced_df[balanced_df['label'] == 0])}")
        
        return balanced_df['title_clean'], balanced_df['label'], None

def train_enhanced_model(X, y, use_smote_data=False, vectorizer=None):
    """
    Train an enhanced fake news detection model
    """
    if not use_smote_data:
        # Split data first, then vectorize
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🎯 Training set size: {len(X_train)}")
        print(f"🎯 Testing set size: {len(X_test)}")
        
        # Initialize TF-IDF Vectorizer with enhanced parameters
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.8,
            min_df=2,
            max_features=8000,
            ngram_range=(1, 3),
            sublinear_tf=True
        )
        
        # Fit and transform the training data
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)
        
    else:
        # Use pre-vectorized SMOTE data
        tfidf_vectorizer = vectorizer
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        tfidf_train = X_train
        tfidf_test = X_test
    
    print(f"📈 TF-IDF feature shape: {tfidf_train.shape}")
    
    # Calculate class weights to handle remaining imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"📊 Class weights: {class_weight_dict}")
    
    # Train multiple models and compare
    models = {}
    
    # 1. PassiveAggressive with class weights
    print(f"\n🤖 Training PassiveAggressive Classifier...")
    pac = PassiveAggressiveClassifier(
        max_iter=100, 
        random_state=42,
        class_weight=class_weight_dict
    )
    pac.fit(tfidf_train, y_train)
    models['PassiveAggressive'] = pac
    
    # 2. Random Forest with class weights
    print(f"🌲 Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight_dict,
        max_depth=10
    )
    rf.fit(tfidf_train, y_train)
    models['RandomForest'] = rf
    
    # Evaluate all models
    best_model = None
    best_score = 0
    best_name = ""
    
    print(f"\n📊 Model Comparison:")
    print("=" * 70)
    
    for name, model in models.items():
        y_pred = model.predict(tfidf_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Use F1-score for fake news as the selection criteria
        from sklearn.metrics import f1_score
        f1_fake = f1_score(y_test, y_pred, pos_label=1)
        
        if f1_fake > best_score:
            best_score = f1_fake
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (F1-score for fake news: {best_score:.4f})")
    
    return best_model, tfidf_vectorizer, best_name

def save_enhanced_model(model, vectorizer, model_name):
    """
    Save the enhanced trained model and vectorizer
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model and vectorizer with enhanced names
        model_filename = f'models/enhanced_fake_news_classifier_{model_name.lower()}.pkl'
        vectorizer_filename = 'models/enhanced_tfidf_vectorizer.pkl'
        
        joblib.dump(model, model_filename)
        joblib.dump(vectorizer, vectorizer_filename)
        
        print(f"\n💾 Enhanced model saved successfully!")
        print(f"- Classifier: {model_filename}")
        print(f"- Vectorizer: {vectorizer_filename}")
        
        return model_filename, vectorizer_filename
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return None, None

def test_enhanced_model(model, vectorizer):
    """
    Test the enhanced model with comprehensive sample predictions
    """
    print(f"\n" + "="*60)
    print("🧪 TESTING ENHANCED MODEL WITH SAMPLE PREDICTIONS")
    print("="*60)
    
    # Enhanced test samples with more variety
    test_samples = [
        # Clearly fake claims
        "COVID-19 vaccines alter your DNA permanently and make you magnetic",
        "5G towers cause coronavirus infection and spread the disease",
        "Drinking bleach or disinfectant cures coronavirus completely",
        "Garlic and ginger can completely prevent COVID-19 infection",
        "The coronavirus is a hoax created by the government",
        
        # Clearly real/factual statements
        "Wearing masks can help reduce the spread of COVID-19",
        "Washing hands frequently helps prevent COVID-19 transmission",
        "COVID-19 vaccines have been tested for safety and efficacy",
        "Social distancing measures can slow the spread of coronavirus",
        "COVID-19 symptoms include fever, cough, and difficulty breathing",
        
        # Borderline/nuanced cases
        "Some people experience mild side effects after COVID-19 vaccination",
        "Natural immunity may provide some protection against COVID-19",
        "Young healthy people are less likely to have severe COVID-19",
    ]
    
    expected_labels = [
        # Expected fake (1)
        1, 1, 1, 1, 1,
        # Expected real (0) 
        0, 0, 0, 0, 0,
        # Nuanced (should be real but could be tricky)
        0, 0, 0
    ]
    
    correct_predictions = 0
    total_predictions = len(test_samples)
    
    for i, (sample, expected) in enumerate(zip(test_samples, expected_labels), 1):
        # Preprocess the sample
        processed_sample = preprocess_text(sample)
        
        # Vectorize the sample
        sample_tfidf = vectorizer.transform([processed_sample])
        
        # Make prediction
        prediction = model.predict(sample_tfidf)[0]
        
        # Get confidence score
        try:
            if hasattr(model, 'decision_function'):
                confidence = abs(model.decision_function(sample_tfidf)[0])
            elif hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(sample_tfidf)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0
        except:
            confidence = 1.0
        
        # Interpret results
        predicted_label = "FAKE" if prediction == 1 else "REAL"
        expected_label = "FAKE" if expected == 1 else "REAL"
        
        # Check if prediction is correct
        is_correct = prediction == expected
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\nTest {i}: {status}")
        print(f"Text: {sample}")
        print(f"Predicted: {predicted_label} | Expected: {expected_label}")
        print(f"Confidence: {confidence:.3f}")
    
    test_accuracy = correct_predictions / total_predictions
    print(f"\n🎯 Test Sample Accuracy: {correct_predictions}/{total_predictions} = {test_accuracy:.2%}")

def main():
    """
    Main enhanced training pipeline
    """
    print("="*70)
    print("🚀 ENHANCED COVID-19 FAKE NEWS DETECTION MODEL TRAINING")
    print("="*70)
    
    # Load multiple datasets
    df = load_multiple_datasets()
    if df is None:
        return
    
    # Try different balancing approaches
    print(f"\n🔄 Testing different balancing approaches...")
    
    # Method 1: Undersampling
    print(f"\n" + "="*50)
    print("📊 METHOD 1: UNDERSAMPLING")
    print("="*50)
    
    X_under, y_under, _ = balance_dataset(df, method='undersample')
    model_under, vectorizer_under, name_under = train_enhanced_model(X_under, y_under)
    
    # Method 2: SMOTE
    if len(df[df['label'] == 1]) >= 6:  # Need at least 6 samples for SMOTE
        print(f"\n" + "="*50)
        print("🔬 METHOD 2: SMOTE OVERSAMPLING")
        print("="*50)
        
        X_smote, y_smote, vec_smote = balance_dataset(df, method='smote')
        model_smote, vectorizer_smote, name_smote = train_enhanced_model(
            X_smote, y_smote, use_smote_data=True, vectorizer=vec_smote
        )
        
        # Choose the best performing model
        print(f"\n🏆 Using undersampling approach for final model")
        final_model = model_under
        final_vectorizer = vectorizer_under
        final_name = name_under
    else:
        print(f"\n⚠️  Insufficient fake samples for SMOTE, using undersampling only")
        final_model = model_under
        final_vectorizer = vectorizer_under 
        final_name = name_under
    
    # Save the final model
    model_file, vec_file = save_enhanced_model(final_model, final_vectorizer, final_name)
    
    # Test with comprehensive samples
    test_enhanced_model(final_model, final_vectorizer)
    
    print(f"\n" + "="*70)
    print(f"🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print(f"✨ Best Model: {final_name}")
    print(f"💾 Model saved as: {model_file}")
    print("="*70)

if __name__ == "__main__":
    main()