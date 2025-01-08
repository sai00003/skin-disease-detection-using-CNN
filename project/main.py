
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('skin_disease_classifier.keras')

# Class labels (replace with your actual class names in the same order as your model’s output)
class_labels = [
    'Acne', 'Atopic Dermatitis Photos', 'Benign Tumors', 'Bullous Disease',
    'Eczema', 'Exanthems', 'Herpes HPV', 'Melanoma Skin Cancer',
    'Nail Fungus', 'Scabies Lyme Disease', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis Photos'
]



def prepare_image(file_path):
    """Load and preprocess image."""
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize
    return image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file and file.filename != '':
            # Save the file temporarily
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            # Prepare the image for the model
            image = prepare_image(file_path)

            # Make prediction
            prediction = model.predict(image)
            predicted_class = class_labels[np.argmax(prediction)]

            # Return results to template
            return render_template('index.html', result=predicted_class,
                                   image=url_for('static', filename=file.filename))

    # Render template for GET requests or when no file is uploaded
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

