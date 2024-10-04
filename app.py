from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Đường dẫn đến các mô hình đã huấn luyện
linear_model_path = "models/linear_model.pkl"
lasso_model_path = "models/lasso_model.pkl"
mlp_model_path = "models/mlp_model.pkl"
stacking_model_path = "models/stacking_model.pkl"
scaler_path = "models/scaler.pkl"

# Tải các mô hình đã lưu
linear_model = joblib.load(linear_model_path)
lasso_model = joblib.load(lasso_model_path)
mlp_model = joblib.load(mlp_model_path)
stacking_model = joblib.load(stacking_model_path)
scaler = joblib.load(scaler_path)

# Trang chủ (hiển thị form nhập liệu)
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý form và dự đoán BMI
@app.route('/prediction', models=['POST'])
def prediction():
    # Lấy dữ liệu từ form
    date = float(request.form['transaction_date'])
    age = float(request.form['house_age'])
    mrt = float(request.form['mrt_distance'])
    store = float(request.form['stores'])
    latitude= float(request.form['longitude'])
    longitude = float(request.form['latitude'])
    price= float(request.form['price'])
    model= request.form['model']
    

    # Tính toán BMI dựa trên chiều cao và cân nặng
    input_data = np.array([[date, age,mrt,store,latitude,longitude,price]])

    # Chuẩn hóa dữ liệu cho Neural Network và Stacking
    input_data_scaled = scaler.transform(input_data)

    # Chọn mô hình dự đoán
    if model == 'LinearRegression':
        prediction = linear_model.predict(input_data)
    elif model == 'lassoRegressionn':
        prediction = lasso_model.predict(input_data)
    elif model == 'NeuralNetwork':
        prediction = mlp_model.predict(input_data_scaled)
    elif model == 'Stacking':
        prediction = stacking_model.predict(input_data_scaled)

    # Làm tròn kết quả dự đoán
    bmi_value = round(prediction[0], 2)

    # Phân loại BMI
    if bmi_value < 18.5:
        bmi_category = "Thiếu cân (Khô lâu đại tướng 💀)"
    elif 18.5 <= bmi_value < 24.99:
        bmi_category = "Cân nặng bình thường (Good 👍)"
    elif 25 <= bmi_value < 29.99:
        bmi_category = "Thừa cân (Fat man 🐽)"
    else:
        bmi_category = "Béo phì (Hốc trưởng 🍴)"

    # Truyền lại các giá trị đã nhập và kết quả dự đoán vào template
    return render_template('index.html', bmi=bmi_value, category=bmi_category,date = date,age = age,mrt = mrt,store = store,latitude = latitude,longitude = longitude,price = price ,model=model)

if __name__ == '__main__':
    app.run(debug=True)