from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
import time
import os

app = Flask(__name__)

# Load model dan label map
model = None
label_map = {}

def load_model():
    global model, label_map
    try:
        # Load model
        model = tf.keras.models.load_model('model_batik.h5')
        print("Model loaded successfully")
        
        # Load label map
        with open('label_map.json', 'r') as f:
            label_map = json.load(f)
        print("Label map loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_file):
    """Preprocess image untuk model"""
    try:
        # Buka gambar
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Convert ke RGB jika diperlukan
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ke 224x224 (sesuaikan dengan model kamu)
        image = image.resize((224, 224))
        
        # Convert ke numpy array
        image_array = np.array(image)
        
        # Normalize pixel values (0-1)
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")

def get_class_name_from_index(index):
    """Convert index ke nama class"""
    for class_name, class_index in label_map.items():
        if class_index == index:
            return class_name
    return f"Unknown_{index}"

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Batik Classifier API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model-info": "/model-info"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded"
            }), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided"
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "error": "No image file selected"
            }), 400
        
        # Preprocess image
        try:
            processed_image = preprocess_image(image_file)
        except Exception as e:
            return jsonify({
                "error": f"Error processing image: {str(e)}"
            }), 400
        
        # Make prediction
        try:
            predictions = model.predict(processed_image)
            predictions_list = predictions[0].tolist()
            
            # Get predicted class
            predicted_index = np.argmax(predictions)
            predicted_class = get_class_name_from_index(predicted_index)
            confidence = float(predictions[0][predicted_index])
            
            # Create all predictions with class names
            all_predictions = []
            for i, prob in enumerate(predictions_list):
                class_name = get_class_name_from_index(i)
                all_predictions.append({
                    "class": class_name,
                    "confidence": float(prob)
                })
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_predictions": all_predictions,
                "processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Error during prediction: {str(e)}"
            }), 500
    
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        return jsonify({
            "input_shape": input_shape,
            "output_shape": output_shape,
            "classes": list(label_map.keys()),
            "num_classes": len(label_map)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error getting model info: {str(e)}"
        }), 500

if __name__ == '__main__':
    try:
        # Load model saat startup
        load_model()
        
        # Railway akan set PORT environment variable
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        exit(1)