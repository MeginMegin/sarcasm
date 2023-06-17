from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

# Load the trained model and TfidfVectorizer
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    new_text = request.form['text']
    new_text_features = tfidf.transform([new_text])
    predicted_class = model.predict(new_text_features)

    if predicted_class == 0:
        prediction = "Not sarcastic."
    else:
        prediction = "Sarcastic!"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
