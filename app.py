from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('m1_rf2.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    hour = int(request.form.get('hour'))
    rainfall = float(request.form.get('rainfall'))
    snowfall = float(request.form.get('snowfall'))
    temperature = float(request.form.get('temperature'))
    dew_point_temperature = float(request.form.get('dew_point_temp'))
    solar_radition  = float(request.form.get('solar_rad'))
    humidity = int(request.form.get('humidity'))
    wind_speed = float(request.form.get('wind_speed'))
    visibility = int(request.form.get('visibility'))
    
    

    # prediction
    result = model.predict(np.array([hour,rainfall,snowfall,temperature,dew_point_temperature,
                                     solar_radition, humidity,wind_speed,visibility,]).reshape(1,9))


    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=8080)
    
