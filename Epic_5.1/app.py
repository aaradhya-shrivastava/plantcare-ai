import os
import secrets
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'images'), exist_ok=True)

# Attempt to load Keras model
model = None

def load_model():
    global model
    print("Loading model...")
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        # Use absolute or relative path where model is expected
        model_path = 'mobilenetv2_best.keras'
        if os.path.exists(model_path):
            model = keras_load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: '{model_path}' not found. Prediction will use mock data.")
    except Exception as e:
        print(f"Warning: Error loading model. Prediction will use mock data. Error: {e}")

def predict_image(filepath):
    global model
    # Generic mock classes for plant disease
    classes = ['Healthy', 'Nitrogen Deficiency', 'Potassium Deficiency', 'Pest Attack', 'Fungal Infection']
    
    if model is not None:
        try:
            from tensorflow.keras.preprocessing.image import img_to_array
            img = Image.open(filepath).convert('RGB')
            # Adjust size to whatever the MobileNetV2 was trained on (typically 224x224)
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            confidence = preds[0][idx] * 100
            
            # Use prediction index if classes mapping not found
            if idx < len(classes):
                predicted_class = classes[idx]
            else:
                predicted_class = f"Class Index {idx}"
                
            return f"{predicted_class} ({confidence:.2f}% certainty)"
        except Exception as e:
            return f"Error predicting image: {str(e)}"
    else:
        # Mock prediction for demo
        import random
        return random.choice(classes) + " (Mock Prediction)"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            prediction = predict_image(filepath)
            
            # Save image to static folder for display
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config['STATIC_FOLDER'], 'images', static_filename)
            Image.open(filepath).save(static_path)
            
            # Store in session
            session['prediction'] = prediction
            session['image_path'] = f'images/{static_filename}'
            
            os.remove(filepath) # Clean up temp file
            
            return jsonify({'success': True})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    
    if not prediction:
        return redirect(url_for('upload'))
        
    return render_template('result.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
