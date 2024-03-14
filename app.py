from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Render the form for input
@app.route('/')
def index():
    return render_template('index.html')

# Perform prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    mood_swing = request.form['Mood Swing']
    optimism = request.form['Optimisim']
    sexual_activity = request.form['Sexual Activity']
    euphoric = request.form['Euphoric']
    suicidal_thoughts = request.form['Suicidal thoughts']
    
    # Convert categorical values to numerical representations
    mood_swing = 1 if mood_swing == 'YES' else 0
    optimism = int(optimism.split()[0])  # Extract the numerical part
    sexual_activity = int(sexual_activity.split()[0])  # Extract the numerical part
    euphoric = 1 if euphoric == 'Often' else (0 if euphoric == 'Seldom' else 0)
    suicidal_thoughts = 1 if suicidal_thoughts == 'YES' else 0
    
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Mood Swing': [mood_swing],
        'Optimisim': [optimism],
        'Sexual Activity': [sexual_activity],
        'Euphoric': [euphoric],
        'Suicidal thoughts': [suicidal_thoughts]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction result
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
