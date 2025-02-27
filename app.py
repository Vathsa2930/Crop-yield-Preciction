from flask import Flask, request, render_template
import numpy as np
import pickle
import plotly.graph_objs as go
import json

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect user inputs
        inputs = {
            "Year": request.form['Year'],
            "average_rain_fall_mm_per_year": request.form['average_rain_fall_mm_per_year'],
            "pesticides_tonnes": request.form['pesticides_tonnes'],
            "avg_temp": request.form['avg_temp'],
            "Area": request.form['Area'],
            "Item": request.form['Item']
        }

        # Prepare features for prediction
        features = np.array([[inputs['Year'], inputs['average_rain_fall_mm_per_year'],
                              inputs['pesticides_tonnes'], inputs['avg_temp'], inputs['Area'], inputs['Item']]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)[0][0]

        # Generate dummy data for trends (replace with real data if available)
        years = list(range(int(inputs["Year"]) - 5, int(inputs["Year"]) + 1))
        predictions = [prediction * (0.9 + 0.02 * i) for i in range(len(years))]

        # Create plotly graph data
        trend_data = json.dumps([{
            "x": years,
            "y": predictions,
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Predictions",
            "line": {"color": "blue"}
        }])

        # Render the template with all required variables
        return render_template(
            'index1.html',
            prediction=prediction,
            inputs=inputs,
            trend_data=trend_data
        )

if __name__ == "__main__":
    app.run(debug=True)
