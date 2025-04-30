from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import tensorflow as tf

from vit_keras import vit  # You might need to install: pip install vit-keras

# ✅ Create a custom loader if needed
vit_model = vit.vit_b16(
    image_size=224,  # Must match original training size
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# ✅ Register the custom object when loading the model
model = load_model('model/model2.h5', custom_objects={'Functional': tf.keras.Model})

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model = load_model('model/model2.h5')



@app.route('/')
def home():
    return render_template('index.html')  # Main page with the form to upload an image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image was uploaded
        if 'image' not in request.files:
            return render_template('index.html', prediction_text="No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction_text="No selected file")
        
        # Open the image file and resize it
        image = Image.open(file.stream)
        image = image.convert('RGB')
        image = image.resize((224, 224))  # Resize to match the input size of the model
        image_array = np.array(image)  # Convert image to a numpy array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Normalize image if needed (depends on how your model was trained)
        image_array = image_array / 255.0
        
        # Predict using the model
        prediction = model.predict(image_array)

        # For binary classification, the model typically outputs a probability.
        # Use a threshold of 0.5 to classify
        predicted_class = "Brain Stroke" if prediction[0] > 0.5 else "No Stroke"

        # Print the prediction probability for debugging
        print(f"Prediction probability: {prediction[0][0]*100:.2f}%")
        prob = prediction[0][0]*100
        # Return the result to the HTML page
        return render_template('index.html', prediction_text=f'Predicted: {predicted_class} <br> Probaility of having stroke:{prob}%')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
if __name__ == '__main__':
    app.run(debug=True)
