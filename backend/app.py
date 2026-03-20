from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
from typing import Dict, List, Tuple
import re
import joblib
from lime.lime_text import LimeTextExplainer
import shap
import pandas as pd

# Import the preprocessing pipeline
from preprocessing_pipeline__1_ import preprocessing_pipeline

# ========================
# Load models at startup
# ========================

app = Flask(__name__)
CORS(app)

# Load vectorizer
vectorizer = joblib.load('models/word_vectorizer.pkl')
background_sample = pd.read_csv('data/sample_data.csv')
X_bg = background_sample['text'].tolist()

# Load models
models_dict = {
    'model_lr': joblib.load('models/word_logistic_model.pkl'),
    'model_svm': joblib.load('models/word_svm_model.pkl'),
    'model_rf': joblib.load('models/word_rf_model.pkl')  # Random Forest
}

# Models output classes in order: ['-', '=', '+'] which maps to ['negative', 'neutral', 'positive']
# We'll create LIME explainer dynamically in the explain_lime function to ensure class order matches

# ========================
# Actual model metadata
# ========================

MODELS = {
    'model_lr': {
        'model': models_dict['model_lr'],
        'name': 'Logistic Regression',
        'type': 'Linear Model',
        'accuracy': 0.93,
        'precision': {'positive': 0.97, 'negative': 0.80, 'neutral': 0.87},
        'recall': {'positive': 0.95, 'negative': 0.85, 'neutral': 0.91},
        'f1': {'positive': 0.96, 'negative': 0.82, 'neutral': 0.89},
    },
    'model_svm': {
        'model': models_dict['model_svm'],  # your calibrated SVM
        'name': 'Word SVM',
        'type': 'SVM',
        'accuracy': 0.92,
        'precision': {'positive': 0.96,'negative': 0.82,'neutral': 0.87},
        'recall': {'positive': 0.96,'negative': 0.78,'neutral': 0.89},
        'f1': {'positive': 0.96,'negative': 0.80,'neutral': 0.88}
    },
    'model_rf': {
        'model': models_dict['model_rf'],
        'name': 'Word Random Forest',
        'type': 'Random Forest',
        'accuracy': 0.90,
        'precision': {'positive': 0.93, 'negative': 0.80, 'neutral': 0.87},
        'recall': {'positive': 0.97, 'negative': 0.68, 'neutral': 0.82},
        'f1': {'positive': 0.95, 'negative': 0.74, 'neutral': 0.85}
    }
}

SENTIMENT_KEYWORDS = []

# ========================
# Utilities
# ========================

def tokenize(text: str) -> List[str]:
    """Simple tokenization"""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

def predict_sentiment(text: str, model_id: str):
    """Real sentiment prediction with preprocessing"""
    model_entry = MODELS[model_id]
    model = model_entry['model']

    # Apply preprocessing pipeline
    processed_text = preprocessing_pipeline(text)
    
    # If preprocessing resulted in empty text, use original
    if not processed_text.strip():
        processed_text = text

    # Ensure text is wrapped in a list
    features = vectorizer.transform([processed_text])

    # Get prediction and probabilities
    prediction = model.predict(features)[0]

    # Some models might not have predict_proba (e.g., LinearSVC)
    try:
        probabilities = model.predict_proba(features)[0]
        # Map probabilities to sentiments based on model.classes_ order
        # model.classes_ is typically ['-', '=', '+']
        class_to_sentiment = {
            '-': 'negative',
            '=': 'neutral',
            '+': 'positive'
        }
        scores = {}
        for idx, class_label in enumerate(model.classes_):
            sentiment_name = class_to_sentiment[class_label]
            scores[sentiment_name] = probabilities[idx]
    except AttributeError:
        # fallback for models without predict_proba
        scores = {'negative': 0, 'neutral': 0, 'positive': 0}

    # Map integer label to sentiment
    label_map = {'-': 'negative', '=': 'neutral', '+': 'positive'}
    sentiment = label_map[prediction]

    return sentiment, scores, processed_text

# ========================
# Explanation functions
# ========================

def explain_shap(text: str, model_id: str, predicted_sentiment: str) -> list:
    """Real SHAP explanation for a single text - works with all model types"""
    model = MODELS[model_id]['model']
    model_type = MODELS[model_id]['type']

    # Preprocess the text
    processed_text = preprocessing_pipeline(text)
    if not processed_text.strip():
        processed_text = text

    # Create explainer with background
    background_features = vectorizer.transform(X_bg)
    
    # Choose appropriate SHAP explainer based on model type
    print(f"SHAP Debug - Model type: {model_type}")
    
    if model_type == 'Linear Model':
        # Use LinearExplainer for logistic regression (fast)
        explainer = shap.LinearExplainer(model, background_features, feature_perturbation="interventional")
    elif model_type == 'SVM':
        # Use KernelExplainer for SVM (slower but works)
        # Sample fewer background points for speed
        background_sample_indices = np.random.choice(background_features.shape[0], min(50, background_features.shape[0]), replace=False)
        background_sample = background_features[background_sample_indices]
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
    elif model_type == 'Random Forest':
        # Use TreeExplainer for Random Forest (fast and accurate)
        explainer = shap.TreeExplainer(model, background_features)
    else:
        # Fallback to KernelExplainer for unknown model types
        background_sample_indices = np.random.choice(background_features.shape[0], min(50, background_features.shape[0]), replace=False)
        background_sample = background_features[background_sample_indices]
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)

    # Transform text
    features = vectorizer.transform([processed_text])
    
    # Compute SHAP values
    shap_values = explainer.shap_values(features)
    
    # Get number of classes from model
    num_classes = len(model.classes_)
    
    # Handle different SHAP value formats
    # TreeExplainer returns list of arrays (one per class)
    # LinearExplainer and KernelExplainer can return different formats
    if isinstance(shap_values, list):
        # Already a list of arrays, one per class
        shap_values_list = shap_values
    elif num_classes > 2 and len(shap_values.shape) == 3:
        # Multi-class: shape is (num_samples, num_features, num_classes)
        # Reshape to list format: one array per class
        shap_values_list = [shap_values[:, :, i] for i in range(num_classes)]
    elif num_classes > 2 and len(shap_values.shape) == 2:
        # Sometimes returns (num_features, num_classes) - need to handle this
        shap_values_list = [shap_values[:, i] for i in range(num_classes)]
    else:
        # Binary: shape is (num_samples, num_features)
        # Create list with negative for class 0, positive for class 1
        shap_values_list = [-shap_values, shap_values]
    
    # Map sentiment to class index using model.classes_
    class_map = {cls: idx for idx, cls in enumerate(model.classes_)}

    sentiment_to_class = {
        'positive': class_map.get('+', 0),
        'neutral':  class_map.get('=', 1),
        'negative': class_map.get('-', 2)
    }
    class_idx = sentiment_to_class[predicted_sentiment]
    
    print(f"SHAP Debug - Model classes: {model.classes_}")
    print(f"SHAP Debug - Predicted sentiment: {predicted_sentiment}")
    print(f"SHAP Debug - Class index: {class_idx}")
    print(f"SHAP Debug - SHAP values type: {type(shap_values)}")
    print(f"SHAP Debug - SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'list'}")

    # Get feature names and create mapping
    feature_names = vectorizer.get_feature_names_out()
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    # Get the SHAP values for the predicted class
    shap_values_for_class = shap_values_list[class_idx]
    
    # Handle both (num_features,) and (1, num_features) shapes
    if len(shap_values_for_class.shape) == 2:
        shap_values_for_class = shap_values_for_class[0]
    
    print(f"SHAP Debug - Values for class shape: {shap_values_for_class.shape}")

    # Extract word-level importance from processed text
    words = processed_text.lower().split()
    
    explanations = []
    for i, word in enumerate(words):
        if word in feature_name_to_idx:
            idx = feature_name_to_idx[word]
            importance = float(shap_values_for_class[idx])
        else:
            importance = 0.0

        explanations.append({
            'word': word,
            'importance': round(importance, 4),
            'position': i
        })

    return explanations


def explain_lime(text: str, model_id: str, predicted_sentiment: str) -> list:
    """Real LIME explanation for a single text - returns contributions for PREDICTED CLASS ONLY"""
    model = MODELS[model_id]['model']

    # Preprocess the text
    processed_text = preprocessing_pipeline(text)
    if not processed_text.strip():
        processed_text = text

    # Get the model's class order
    model_classes = model.classes_  # Should be ['-', '=', '+']
    
    # Create class names in the same order as model.classes_
    # Map ['-', '=', '+'] to ['negative', 'neutral', 'positive']
    class_name_map = {
        '-': 'negative',
        '=': 'neutral',
        '+': 'positive'
    }
    class_names_ordered = [class_name_map[cls] for cls in model_classes]
    
    # Create LIME explainer with correct class order for this model
    lime_explainer = LimeTextExplainer(class_names=class_names_ordered)
    
    def predict_fn(texts):
        # Preprocess all texts in the batch
        processed_texts = [preprocessing_pipeline(t) if preprocessing_pipeline(t).strip() else t for t in texts]
        features = vectorizer.transform(processed_texts)
        proba = model.predict_proba(features)
        return proba

    # Generate explanation
    exp = lime_explainer.explain_instance(
        processed_text,
        predict_fn,
        num_features=20,
        num_samples=500  # Add explicit num_samples for stability
    )

    # Map sentiment to index in class_names_ordered
    predicted_class_idx = class_names_ordered.index(predicted_sentiment)
    
    print(f"LIME Debug - Model classes: {model_classes}")
    print(f"LIME Debug - Class names ordered: {class_names_ordered}")
    print(f"LIME Debug - Predicted sentiment: {predicted_sentiment}")
    print(f"LIME Debug - Predicted class idx: {predicted_class_idx}")
    print(f"LIME Debug - Available labels: {exp.available_labels()}")
    
    # Get ONLY the contributions for the predicted class
    # as_list() returns tuples of (feature, weight) for the specified label
    try:
        word_scores_for_predicted_class = dict(exp.as_list(label=predicted_class_idx))
        print(f"LIME Debug - Got {len(word_scores_for_predicted_class)} features for class {predicted_class_idx}")
    except Exception as e:
        print(f"Error getting LIME explanations for class {predicted_class_idx}: {e}")
        print(f"Falling back to default explanation")
        # Fallback: use the default (highest probability class)
        word_scores_for_predicted_class = dict(exp.as_list())

    # Extract word importance for each word in the processed text
    words = processed_text.lower().split()
    
    explanations = []
    for i, word in enumerate(words):
    # Get the importance for this word (0.0 if not found)
        importance = float(word_scores_for_predicted_class.get(word, 0.0))

        if predicted_sentiment in ['positive', 'neutral']:
            importance = -importance

        explanations.append({
            'word': word,
            'importance': round(importance, 4),
            'position': i
        })

    return explanations

# ========================
# API Endpoints
# ========================

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for sentiment prediction"""
    try:
        data = request.json
        text = data.get('text', '')
        model_ids = data.get('models', list(MODELS.keys()))
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        
        results = []
        for model_id in model_ids:
            if model_id not in MODELS:
                continue
            sentiment, scores, processed_text = predict_sentiment(text, model_id)
            results.append({
                'model_id': model_id,
                'model_name': MODELS[model_id]['name'],
                'sentiment': sentiment,
                'confidence': round(scores[sentiment], 4),
                'scores': {k: round(v, 4) for k, v in scores.items()},
                'preprocessed_text': processed_text  # Include preprocessed text in response
            })
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    """Endpoint for generating explanations with debug prints"""
    try:
        data = request.json
        print("Received JSON:", data)

        text = data.get('text', '')
        print("Text received:", text)

        model_id = data.get('model', list(MODELS.keys())[0])
        print("Model ID received:", model_id)

        methods = data.get('methods', ['shap'])
        print("Methods requested:", methods)

        # Validate input
        if not text:
            print("Error: No text provided")
            return jsonify({'error': 'No text provided'}), 400
        if model_id not in MODELS:
            print(f"Error: Invalid model {model_id}")
            return jsonify({'error': f'Invalid model: {model_id}'}), 400

        # Predict sentiment (now returns preprocessed text too)
        sentiment, scores, processed_text = predict_sentiment(text, model_id)
        print("Predicted sentiment:", sentiment)
        print("Preprocessed text:", processed_text)
        print("Model classes:", MODELS[model_id]['model'].classes_)

        explanations = {}
        if 'shap' in methods:
            print("Generating SHAP explanations...")
            try:
                explanations['shap'] = explain_shap(text, model_id, sentiment)
                print(f"SHAP explanations generated: {len(explanations['shap'])} words")
            except Exception as e:
                print(f"Error generating SHAP explanations: {e}")
                import traceback
                traceback.print_exc()
                
        if 'lime' in methods:
            print("Generating LIME explanations...")
            try:
                explanations['lime'] = explain_lime(text, model_id, sentiment)
                print(f"LIME explanations generated: {len(explanations['lime'])} words")
            except Exception as e:
                print(f"Error generating LIME explanations: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'LIME explanation failed: {str(e)}'}), 500

        print("Explanations generated successfully")
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'preprocessed_text': processed_text,  # Include in response
            'explanations': explanations
        })

    except Exception as e:
        print("Exception in /explain:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/report', methods=['POST'])
def report():
    """Endpoint for getting classification report"""
    try:
        data = request.json
        model_ids = data.get('models', list(MODELS.keys()))
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        
        reports = []
        for model_id in model_ids:
            if model_id not in MODELS:
                continue
            m = MODELS[model_id]
            reports.append({
                'model_id': model_id,
                'model_name': m['name'],
                'model_type': m['type'],
                'accuracy': m['accuracy'],
                'metrics': {
                    'positive': {'precision': m['precision']['positive'], 'recall': m['recall']['positive'], 'f1-score': m['f1']['positive']},
                    'negative': {'precision': m['precision']['negative'], 'recall': m['recall']['negative'], 'f1-score': m['f1']['negative']},
                    'neutral': {'precision': m['precision']['neutral'], 'recall': m['recall']['neutral'], 'f1-score': m['f1']['neutral']}
                }
            })
        
        return jsonify({'success': True, 'reports': reports})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Endpoint for testing preprocessing on text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        processed_text = preprocessing_pipeline(text)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'preprocessed_text': processed_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': len(MODELS)})

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Loaded {len(MODELS)} models: {', '.join(MODELS.keys())}")
    print("Preprocessing pipeline integrated successfully!")
    app.run(debug=True, port=5000)