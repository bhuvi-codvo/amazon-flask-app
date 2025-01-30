import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from tensorflow.keras.models import load_model
from pathlib import Path
from flask import Flask, request, jsonify

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

class AmazonFurniturePredictor:
    def __init__(self):
        try:
            # Load model using relative paths
            base_path = Path(__file__).parent / "models"
            self.model = load_model(base_path / "model_amazon.keras")
            
            # Load tokenizer
            with open(base_path / "tokenizer_amazon.pickle", 'rb') as handle:
                self.tokenizer = pickle.load(handle)
                
            # Initialize preprocessing components
            self.nltk_tokenizer = RegexpTokenizer(r"[\w']+")
            self.stop_words = set(stopwords.words('english'))
            
            # Load category and leasability mappings
            self.categories_list = pd.read_csv(base_path / "amazon_categories.csv")["0"].tolist()
            self.leasability_mapping = self._load_leasability_mapping()
        except Exception as e:
            print(f"Error initializing predictor: {str(e)}")
            raise
    
    def _load_leasability_mapping(self):
        """Load leasability mapping from CSV"""
        base_path = Path(__file__).parent / "models"
        leasibility_df = pd.read_csv(base_path / "Amazon Leasibility Altered.csv")
        leasibility_df['cat'] = leasibility_df['cat'].str.replace("'", "").str.replace('"', '')
        return dict(zip(leasibility_df['cat'], leasibility_df['isLeasable']))
    
    def preprocess_text(self, texts):
        """Preprocess multiple texts"""
        try:
            data = pd.Series(texts)
            
            # Use the class tokenizer
            nltk_tokenizer = self.nltk_tokenizer
            stop_words = self.stop_words
            
            # Simplified preprocessing steps
            data = data.apply(lambda x: nltk_tokenizer.tokenize(str(x).lower()))
            data = data.apply(lambda x: ' '.join([w for w in x if w not in stop_words]))
            
            # Convert to sequences using the loaded tokenizer
            X_seq = self.tokenizer.texts_to_sequences(data.values)
            X = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=25)
            return X
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def predict_bulk(self, texts):
        """
        Predict category and leasability for multiple texts
        Args:
            texts: List of product descriptions
        Returns: List of predictions
        """
        try:
            results = []
            
            # Preprocess all texts at once
            processed_texts = self.preprocess_text(texts)
            
            # Get model predictions for all texts
            predictions_array = self.model.predict(processed_texts)
            predicted_category_idx = np.argmax(predictions_array, axis=1)
            
            # Create predictions for each text
            for idx, text in enumerate(texts):
                category = self.categories_list[predicted_category_idx[idx]]
                confidence_score = float(predictions_array[idx][predicted_category_idx[idx]])
                
                # Get leasability
                leasability_value = self.leasability_mapping.get(category, False)
                if isinstance(leasability_value, (bool, np.bool_)):
                    leasability_value = 1 if leasability_value else 0
                
                results.append({
                    "product_text": str(text),
                    "predicted_category": str(category),
                    "predicted_leasability": int(leasability_value),
                    "confidence_score": float(confidence_score)
                })
            
            return results
        except Exception as e:
            print(f"Error in bulk prediction: {str(e)}")
            raise

# Add a health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"})

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        products = data.get('products', [])
        
        if not products:
            return jsonify({"error": "No products provided"}), 400
        
        if not isinstance(products, list):
            return jsonify({"error": "Products must be provided as a list"}), 400
            
        # Initialize predictor
        predictor = AmazonFurniturePredictor()
        
        # Get predictions for all products
        results = predictor.predict_bulk(products)
        
        return jsonify({
            "status": "success",
            "total_products": len(products),
            "predictions": results
        })
        
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 