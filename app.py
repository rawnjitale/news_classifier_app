from flask import Flask, request, jsonify, render_template
import joblib

# Load the model and vectorizer
model = joblib.load('news_classifier_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')
newsgroups = joblib.load('news_group.pkl')  # Save newsgroups data separately if needed

# Initialize Flask app
app = Flask(__name__)

# Route for the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and predict category
@app.route('/predict', methods=['POST'])
def predict():
    # Get text input from form
    text = request.form['text']

    # Transform text using the loaded vectorizer
    text_bow = vectorizer.transform([text])

    # Predict the category
    predicted_index = model.predict(text_bow)[0]
    category = newsgroups.target_names[predicted_index]
    category=category.split('.')
    # Render result on the HTML page
    return render_template('index.html', prediction=category)

if __name__ == '__main__':
    app.run(debug=True)
