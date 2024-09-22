from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
import pandas as pd

# Load the dataset
df = pd.read_csv('styles.csv')

# Define the mapping for the 7 categories
category_mapping = {
    'Shirts': 'Apparel', 'T-Shirts': 'Apparel', 'Blouses': 'Apparel',
    'Jeans': 'Apparel', 'Trousers': 'Apparel', 'Skirts': 'Apparel',
    'Sneakers': 'Footwear', 'Sandals': 'Footwear', 'Shoes': 'Footwear',
    'Belts': 'Accessories', 'Hats': 'Accessories', 'Scarves': 'Accessories',
    'Watches': 'Accessories',
    'Earrings': 'Jewelry', 'Necklaces': 'Jewelry', 'Bracelets': 'Jewelry',
    'Lipstick': 'Personal Care', 'Skincare': 'Personal Care',
}

# Apply the mapping
df['masterCategory'] = df['articleType'].map(category_mapping).fillna('Other')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

# Define the upload folder path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained model
model = load_model('model.h5')

# Class names mapping
class_names = {
    0: 'Apparel',
    1: 'Accessories',
    2: 'Footwear',
    3: 'Personal Care',
    4: 'Free Items',
    5: 'Sporting Goods',
    6: 'Home'
}

# Utility function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize image array if required by the model
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    
    # Get the predicted class
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names.get(predicted_class_index, "Unknown")
    
    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No files part')
            return redirect(request.url)
        files = request.files.getlist('files')
        if len(files) == 0:
            flash('No selected files')
            return redirect(request.url)

        # Initialize category counts
        category_count = {
            'Apparel': 0,
            'Accessories': 0,
            'Footwear': 0,
            'Personal Care': 0,
            'Free Items': 0,
            'Sporting Goods': 0,
            'Home': 0,
            'Unknown': 0
        }

        # Process each file
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Classify the image
                classification = predict_image(file_path)
                
                # Increment the count for the predicted category
                if classification in category_count:
                    category_count[classification] += 1
                else:
                    category_count['Unknown'] += 1
        
        # Flash messages for each category
        for category, count in category_count.items():
            if count > 0:
                flash(f"{count} images classified as {category}", "info")
        
        return redirect(url_for('classify'))
    
    return render_template('classify.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/reviews')  # Ensure this is unique
def reviews():
    return render_template('reviews.html')

@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
