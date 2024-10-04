from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Đường dẫn đến các mô hình đã huấn luyện
linear_model_path = "models/linear_model.pkl"
ridge_model_path = "models/ridge_model.pkl"
mlp_model_path = "models/mlp_model.pkl"
stacking_model_path = "models/stacking_model.pkl"
scaler_path = "models/scaler.pkl"

# Tải các mô hình đã lưu
linear_model = joblib.load(linear_model_path)
ridge_model = joblib.load(ridge_model_path)
mlp_model = joblib.load(mlp_model_path)
stacking_model = joblib.load(stacking_model_path)
scaler = joblib.load(scaler_path)

# Trang chủ (hiển thị form nhập liệu)
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý form và dự đoán BMI
@app.route('/prediction', methods=['POST'])
def prediction():
    # Lấy dữ liệu từ form
    weight = float(request.form['Feature1'])
    height = float(request.form['Feature2'])
    method = request.form['method']

    # Tính toán BMI dựa trên chiều cao và cân nặng
    input_data = np.array([[height, weight]])

    # Chuẩn hóa dữ liệu cho Neural Network và Stacking
    input_data_scaled = scaler.transform(input_data)

    # Chọn mô hình dự đoán
    if method == 'LinearRegression':
        prediction = linear_model.predict(input_data)
    elif method == 'RidgeRegressionn':
        prediction = ridge_model.predict(input_data)
    elif method == 'NeuralNetwork':
        prediction = mlp_model.predict(input_data_scaled)
    elif method == 'Stacking':
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
    return render_template('index.html', bmi=bmi_value, category=bmi_category, weight=weight, height=height, method=method)

if __name__ == '__main__':
    app.run(debug=True)