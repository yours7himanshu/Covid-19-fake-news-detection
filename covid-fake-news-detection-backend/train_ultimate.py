import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def advanced_preprocess_text(text):
    """
    Advanced text preprocessing with multiple techniques
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

def load_comprehensive_datasets():
    """
    Load ALL available COVID-19 datasets comprehensively
    """
    print("🔍 DISCOVERING ALL AVAILABLE DATASETS...")
    print("=" * 60)
    
    all_data = []
    dataset_stats = []
    
    # Get all CSV files in datasets directory
    all_files = glob.glob("./datasets/*.csv")
    
    for file_path in all_files:
        try:
            print(f"📁 Processing: {os.path.basename(file_path)}")
            
            # Determine if it's fake or real from filename
            is_fake = 'Fake' in file_path
            label = 1 if is_fake else 0
            category = 'fake' if is_fake else 'real'
            
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Extract text data based on available columns
            text_data = []
            text_source = []
            
            # Priority order for text extraction
            if 'content' in df.columns and df['content'].notna().sum() > 0:
                # Use full content if available (best quality)
                text_data = df['content'].fillna('')
                text_source = 'content'
                
            elif 'abstract' in df.columns and df['abstract'].notna().sum() > 0:
                # Use abstract if content not available
                text_data = df['abstract'].fillna('')
                text_source = 'abstract'
                
            elif 'title' in df.columns and df['title'].notna().sum() > 0:
                # Use title as fallback
                text_data = df['title'].fillna('')
                text_source = 'title'
                
            elif 'newstitle' in df.columns and df['newstitle'].notna().sum() > 0:
                # Alternative title column
                text_data = df['newstitle'].fillna('')
                text_source = 'newstitle'
            
            else:
                print(f"  ⚠️  No usable text columns found, skipping...")
                continue
            
            # Create processed dataset
            if len(text_data) > 0:
                processed_df = pd.DataFrame({
                    'text': text_data,
                    'label': label,
                    'category': category,
                    'source_file': os.path.basename(file_path),
                    'text_source': text_source
                })
                
                # Remove empty texts
                processed_df = processed_df[processed_df['text'].str.len() > 10]
                
                if len(processed_df) > 0:
                    all_data.append(processed_df)
                    dataset_stats.append({
                        'file': os.path.basename(file_path),
                        'samples': len(processed_df),
                        'category': category,
                        'text_source': text_source
                    })
                    print(f"  ✅ Added {len(processed_df)} samples from {text_source}")
                else:
                    print(f"  ⚠️  All texts too short, skipping...")
            else:
                print(f"  ⚠️  No text data found, skipping...")
                
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
    
    # Combine all datasets
    if not all_data:
        print("❌ No datasets loaded successfully!")
        return None, None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Advanced preprocessing
    print(f"\n🔄 Applying advanced text preprocessing...")
    combined_df['text_processed'] = combined_df['text'].apply(advanced_preprocess_text)
    
    # Remove texts that became empty after preprocessing
    combined_df = combined_df[combined_df['text_processed'].str.len() > 5]
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text_processed'], keep='first')
    dedupe_count = len(combined_df)
    
    print(f"📊 COMPREHENSIVE DATASET SUMMARY:")
    print("=" * 60)
    print(f"Total files processed: {len(dataset_stats)}")
    print(f"Total samples collected: {initial_count}")
    print(f"After deduplication: {dedupe_count}")
    print(f"Fake news samples: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Real news samples: {len(combined_df[combined_df['label'] == 0])}")
    
    print(f"\n📋 DATASET BREAKDOWN:")
    stats_df = pd.DataFrame(dataset_stats)
    for category in ['fake', 'real']:
        cat_stats = stats_df[stats_df['category'] == category]
        print(f"\n{category.upper()} NEWS:")
        for _, row in cat_stats.iterrows():
            print(f"  {row['file']}: {row['samples']} samples ({row['text_source']})")
    
    return combined_df, stats_df

def create_advanced_features(texts):
    """
    Create multiple feature representations
    """
    print(f"🔧 Creating advanced feature representations...")
    
    # TF-IDF with different parameters
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=5000,
        stop_words='english'
    )
    
    tfidf_word = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=10000,
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    
    # Fit transformers
    char_features = tfidf_char.fit_transform(texts)
    word_features = tfidf_word.fit_transform(texts)
    
    # Combine features
    from scipy.sparse import hstack
    combined_features = hstack([char_features, word_features])
    
    print(f"  📈 Character n-grams: {char_features.shape}")
    print(f"  📈 Word n-grams: {word_features.shape}")
    print(f"  📈 Combined features: {combined_features.shape}")
    
    return combined_features, tfidf_char, tfidf_word

def balance_dataset_advanced(X, y, method='smoteenn'):
    """
    Advanced dataset balancing with multiple techniques
    """
    print(f"\n⚖️  BALANCING DATASET WITH {method.upper()}...")
    
    fake_count = sum(y == 1)
    real_count = sum(y == 0)
    
    print(f"Original distribution:")
    print(f"  Fake: {fake_count}")
    print(f"  Real: {real_count}")
    print(f"  Ratio: 1:{real_count/fake_count:.1f}")
    
    if method == 'smoteenn':
        # SMOTE + Tomek (both over and under sampling)
        smote_tomek = SMOTETomek(random_state=42)
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
    elif method == 'smote':
        # Only SMOTE oversampling
        smote = SMOTE(random_state=42, k_neighbors=min(5, fake_count-1))
        X_balanced, y_balanced = smote.fit_resample(X, y)
    else:
        # No balancing
        X_balanced, y_balanced = X, y
    
    fake_balanced = sum(y_balanced == 1)
    real_balanced = sum(y_balanced == 0)
    
    print(f"After balancing:")
    print(f"  Fake: {fake_balanced}")
    print(f"  Real: {real_balanced}")
    print(f"  Ratio: 1:{real_balanced/fake_balanced:.1f}")
    
    return X_balanced, y_balanced

def train_ensemble_model(X, y):
    """
    Train multiple models and create an ensemble
    """
    print(f"\n🤖 TRAINING ADVANCED ENSEMBLE MODEL...")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Define models
    models = {
        'PassiveAggressive': PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=42,
            class_weight=class_weight_dict
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=class_weight_dict,
            solver='liblinear'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weight_dict,
            max_depth=15,
            min_samples_split=5
        ),
        'SVM': SVC(
            kernel='linear',
            random_state=42,
            class_weight=class_weight_dict,
            probability=True
        )
    }
    
    # Train and evaluate individual models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_fake = f1_score(y_test, y_pred, pos_label=1)
        f1_real = f1_score(y_test, y_pred, pos_label=0)
        
        model_results[name] = {
            'accuracy': accuracy,
            'f1_fake': f1_fake,
            'f1_real': f1_real,
            'f1_avg': (f1_fake + f1_real) / 2
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Fake: {f1_fake:.4f}")
        print(f"  F1-Real: {f1_real:.4f}")
    
    # Create ensemble
    print(f"\n🎯 Creating ensemble model...")
    
    # Select top 3 models based on average F1 score
    sorted_models = sorted(model_results.items(), 
                          key=lambda x: x[1]['f1_avg'], 
                          reverse=True)[:3]
    
    ensemble_models = [(name, trained_models[name]) for name, _ in sorted_models]
    
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='hard'  # Use hard voting since PassiveAggressive doesn't have predict_proba
    )
    
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_f1_fake = f1_score(y_test, y_pred_ensemble, pos_label=1)
    ensemble_f1_real = f1_score(y_test, y_pred_ensemble, pos_label=0)
    
    print(f"\n🏆 ENSEMBLE RESULTS:")
    print(f"  Models used: {[name for name, _ in ensemble_models]}")
    print(f"  Accuracy: {ensemble_accuracy:.4f}")
    print(f"  F1-Fake: {ensemble_f1_fake:.4f}")
    print(f"  F1-Real: {ensemble_f1_real:.4f}")
    
    # Detailed classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred_ensemble, target_names=['Real', 'Fake']))
    
    # Find best individual model
    best_individual = max(model_results.items(), key=lambda x: x[1]['f1_avg'])
    best_model_name = best_individual[0]
    best_model = trained_models[best_model_name]
    
    return ensemble, best_model, best_model_name, ensemble_accuracy

def save_ultimate_model(ensemble_model, best_individual, char_vectorizer, word_vectorizer, model_name):
    """
    Save the ultimate model and all components
    """
    try:
        os.makedirs('models', exist_ok=True)
        
        # Save ensemble model
        ensemble_file = 'models/ultimate_ensemble_classifier.pkl'
        joblib.dump(ensemble_model, ensemble_file)
        
        # Save best individual model
        individual_file = f'models/ultimate_best_{model_name.lower()}.pkl'
        joblib.dump(best_individual, individual_file)
        
        # Save vectorizers
        char_vec_file = 'models/ultimate_char_vectorizer.pkl'
        word_vec_file = 'models/ultimate_word_vectorizer.pkl'
        
        joblib.dump(char_vectorizer, char_vec_file)
        joblib.dump(word_vectorizer, word_vec_file)
        
        print(f"\n💾 ULTIMATE MODEL SAVED!")
        print(f"  Ensemble: {ensemble_file}")
        print(f"  Best Individual: {individual_file}")
        print(f"  Character Vectorizer: {char_vec_file}")
        print(f"  Word Vectorizer: {word_vec_file}")
        
        return ensemble_file, individual_file, char_vec_file, word_vec_file
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None, None, None, None

def test_ultimate_model(model, char_vec, word_vec):
    """
    Test the ultimate model with comprehensive examples
    """
    print(f"\n" + "="*70)
    print("🧪 TESTING ULTIMATE MODEL WITH COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    test_cases = [
        # Clearly fake scientific claims
        ("COVID-19 vaccines alter your DNA permanently and make you magnetic", "FAKE"),
        ("5G towers cause coronavirus infections by weakening your immune system", "FAKE"),
        ("Drinking bleach or disinfectant completely cures coronavirus", "FAKE"),
        ("The coronavirus was created in a lab as a biological weapon", "FAKE"),
        ("Hydroxychloroquine is 100% effective against COVID-19 with no side effects", "FAKE"),
        ("Face masks actually make you sicker by trapping carbon dioxide", "FAKE"),
        ("Hot weather and sunshine completely kill the coronavirus", "FAKE"),
        
        # Clearly real scientific facts
        ("Wearing masks can significantly reduce the spread of COVID-19", "REAL"),
        ("Washing hands frequently with soap helps prevent COVID-19 transmission", "REAL"),
        ("COVID-19 vaccines have been tested for safety and efficacy in clinical trials", "REAL"),
        ("Social distancing measures can help slow the spread of coronavirus", "REAL"),
        ("COVID-19 symptoms commonly include fever, cough, and difficulty breathing", "REAL"),
        ("Vaccines work by training your immune system to recognize and fight viruses", "REAL"),
        ("The coronavirus can spread through respiratory droplets when people cough or sneeze", "REAL"),
        
        # Nuanced/borderline cases
        ("Some people may experience mild side effects after COVID-19 vaccination", "REAL"),
        ("Young healthy people are generally less likely to have severe COVID-19 symptoms", "REAL"),
        ("Natural immunity from previous infection may provide some protection against COVID-19", "REAL"),
        ("Long-term effects of COVID-19 are still being studied by researchers", "REAL"),
        ("Vitamin D deficiency may be associated with increased COVID-19 risk", "REAL"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        # Preprocess text
        processed_text = advanced_preprocess_text(text)
        
        # Create features
        char_features = char_vec.transform([processed_text])
        word_features = word_vec.transform([processed_text])
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([char_features, word_features])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        predicted_label = "FAKE" if prediction == 1 else "REAL"
        
        # Get confidence
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(combined_features)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0
        except:
            confidence = 1.0
        
        # Check correctness
        is_correct = predicted_label == expected
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n{status} Test {i:2d}: {predicted_label:4s} (expected {expected:4s}) | Confidence: {confidence:.3f}")
        print(f"         Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    
    accuracy = correct / total
    print(f"\n🎯 ULTIMATE MODEL TEST ACCURACY: {correct}/{total} = {accuracy:.1%}")
    
    return accuracy

def main():
    """
    Ultimate training pipeline using ALL available data
    """
    print("="*80)
    print("🚀 ULTIMATE COVID-19 FAKE NEWS DETECTION MODEL")
    print("🎯 Using ALL Available Datasets with Advanced Techniques")
    print("="*80)
    
    # Load all datasets comprehensively
    df, stats = load_comprehensive_datasets()
    if df is None:
        return
    
    # Create advanced features
    features, char_vec, word_vec = create_advanced_features(df['text_processed'])
    
    # Balance dataset
    X_balanced, y_balanced = balance_dataset_advanced(features, df['label'], method='smoteenn')
    
    # Train ensemble model
    ensemble_model, best_individual, best_name, accuracy = train_ensemble_model(X_balanced, y_balanced)
    
    # Save models
    save_ultimate_model(ensemble_model, best_individual, char_vec, word_vec, best_name)
    
    # Test the model
    test_accuracy = test_ultimate_model(ensemble_model, char_vec, word_vec)
    
    print(f"\n" + "="*80)
    print(f"🎉 ULTIMATE MODEL TRAINING COMPLETED!")
    print(f"📊 Training Accuracy: {accuracy:.1%}")
    print(f"🎯 Test Accuracy: {test_accuracy:.1%}")
    print(f"🏆 Best Individual Model: {best_name}")
    print(f"💡 Total Samples Used: {len(df)}")
    print("="*80)

if __name__ == "__main__":
    main()