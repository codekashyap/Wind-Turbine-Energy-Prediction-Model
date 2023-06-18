from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
model1 = pickle.load(open('linear_model.pkl', 'rb'))
model2 = pickle.load(open('forest_model.pkl', 'rb'))
model3 = pickle.load(open('xgb_model.pkl', 'rb'))
detail = pickle.load(open('detail.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def pred():
    wind_speed = float(request.form['wind_speed'])
    wind_direction = float(request.form['wind_direction'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    hour = int(request.form['hour'])
    final_features = [[wind_speed, wind_direction, month, day, hour]]
    a = pd.DataFrame(final_features, columns=['Wind Speed (m/s)', 'Wind Direction (°)', 'MONTH', 'DAY', 'hour'])
    prediction1 = model1.predict(final_features)
    prediction2 = model2.predict(final_features)
    prediction3 = model3.predict(a)

    output1 = round(prediction1[0], 2)
    output2 = round(prediction2[0], 2)
    output3 = round(prediction3[0], 2)
    detail1 = round(detail[0], 2)
    detail2 = round(detail[1], 2)
    detail3 = round(detail[2], 2)
    detail4 = round(detail[3], 2)
    detail5 = round(detail[4], 2)
    detail6 = round(detail[5], 2)

    return render_template('index.html', output1=output1, output2=output2, output3=output3, detail1=detail1,
                           detail2=detail2, detail3=detail3, detail4=detail4, detail5=detail5, detail6=detail6)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
