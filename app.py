from flask import Flask, render_template, request
from pandas.core.arrays.interval import IntervalSide

from src.pipe import prediction_pipeline
from src.pipe.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['POST', 'GET'])
def form():
    if request.method == "POST":

        # Personal Information
        age = request.form['age']
        gender = request.form['gender']
        height_cm = request.form['height_cm']
        weight_kg = request.form['weight_kg']
        bmi = request.form['bmi']

        # Activity Details
        activity_type = request.form['activity_type']
        duration_minutes = request.form['duration_minutes']
        intensity = request.form['intensity']
        calories_burned = request.form['calories_burned']

        # Health Metrics
        avg_heart_rate = request.form['avg_heart_rate']
        resting_heart_rate = request.form['resting_heart_rate']
        blood_pressure_systolic = request.form['blood_pressure_systolic']
        blood_pressure_diastolic = request.form['blood_pressure_diastolic']

        # Lifestyle Factors
        hours_sleep = request.form['hours_sleep']
        stress_level = request.form['stress_level']
        daily_steps = request.form['daily_steps']
        hydration_level = request.form['hydration_level']
        smoking_status = request.form['smoking_status']

        custom_data = CustomData(
            age, 
            gender,
            height_cm,
            weight_kg,
            bmi,
            activity_type,
            duration_minutes,
            intensity,
            calories_burned,
            avg_heart_rate,
            resting_heart_rate,
            blood_pressure_systolic,
            blood_pressure_diastolic,
            hours_sleep,
            stress_level,
            daily_steps,
            hydration_level,
            smoking_status
        )

        input_querie = custom_data.get_data_as_dataframe()

        predict_pipe = PredictionPipeline()
        prediction = predict_pipe.predict(input_querie)

        return render_template('form.html', prediction=prediction)
    
    else:
        return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)