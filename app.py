from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
import time
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables
model = None
label_map = {}
metadata = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global model, label_map, metadata
    try:
        # Load model
        model_path = os.path.join(os.getcwd(), 'model_batik.h5')
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load label map
        with open('label_map.json', 'r') as f:
            label_map = json.load(f)
        logger.info("Label map loaded successfully")
        
        # Load metadata
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        logger.info("Metadata loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def preprocess_image(image_file):
    """Preprocess image untuk model"""
    try:
        # Check file size
        image_file.seek(0, 2)  # Seek to end
        file_size = image_file.tell()
        image_file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise Exception("File size too large (max 16MB)")
        
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time(),
        "version": "1.0.0"
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
        
        # Check file extension
        if not allowed_file(image_file.filename):
            return jsonify({
                "error": "Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP"
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
            
            # Get metadata for predicted class
            class_metadata = metadata.get(predicted_class, {})
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "metadata": class_metadata,
                "all_predictions": all_predictions[:5],  # Top 5 only
                "processing_time": processing_time
            })
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return jsonify({
                "error": f"Error during prediction: {str(e)}"
            }), 500
    
    except Exception as e:
        logger.error(f"Internal server error: {e}")
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
            "num_classes": len(label_map),
            "model_summary": str(model.summary)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error getting model info: {str(e)}"
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available batik classes with metadata"""
    return jsonify({
        "classes": metadata,
        "total_classes": len(metadata)
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    try:
        # Load model saat startup
        load_model()
        
        # Run server
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_ENV') != 'production'
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)