from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Tải mô hình
model_dir = 'models'
linear_model_path = os.path.join(model_dir, 'linear_model.pkl')
lasso_model_path = os.path.join(model_dir, 'lasso_model.pkl')
mlp_model_path = os.path.join(model_dir, 'nn_model.pkl')
stacking_model_path = os.path.join(model_dir, 'stacking_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

# Kiểm tra sự tồn tại của các tệp mô hình và scaler
for model_path in [linear_model_path, lasso_model_path, mlp_model_path, stacking_model_path, scaler_path]:
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")

# Tải các mô hình và scaler
linear_model = joblib.load(linear_model_path)
lasso_model = joblib.load(lasso_model_path)
mlp_model = joblib.load(mlp_model_path)
stacking_model = joblib.load(stacking_model_path)
scaler = joblib.load(scaler_path)

print("Models and scaler loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['house_age'] or 0 )
        distance = float(request.form['mrt_distance'] or 0)
        store = float(request.form['stores'] or 0)
        latitude = float(request.form['latitude'] or 0)
        model_choice = request.form['model']
        
        # In các giá trị đầu vào để kiểm tra
        print("Input Values - Age:", age, "Distance:", distance, "Store:", store, "Latitude:", latitude)

        # Tạo mảng đầu vào và chuẩn hóa
        features = np.array([[age, distance, store, latitude]])
        scaled_features = scaler.transform(features)

        if model_choice == 'linear':
            prediction = linear_model.predict(scaled_features)[0]
        elif model_choice == 'lasso':
            prediction = lasso_model.predict(scaled_features)[0]
        elif model_choice == 'nn':
            prediction = mlp_model.predict(scaled_features)[0]
        elif model_choice == 'stacking':
            prediction = stacking_model.predict(scaled_features)[0]
            
        if prediction < 0:
                prediction = "Không thể dự đoán được giá bất động sản "
        else:
            prediction = f"Giá bất động sản dự đoán / m2 là: {prediction:.2f}"

        return render_template('index.html', prediction=prediction)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
