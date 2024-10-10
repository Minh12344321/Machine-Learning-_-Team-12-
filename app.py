from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

file_path1 = 'models/linear.pkl'
file_path2 = 'models/lasso.pkl'
file_path3 = 'models/mlp.pkl'
file_path4 = 'models/stacking.pkl'

# Kiểm tra xem các file mô hình có tồn tại không
for file_path in [file_path1, file_path2, file_path3, file_path4]:
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist!")

model = joblib.load(file_path1)
model2 = joblib.load(file_path2)
model3 = joblib.load(file_path3)
model4 = joblib.load(file_path4)

# Kiểm tra loại của từng mô hình
print(f"Model type (Linear): {type(model)}")
print(f"Model type (Lasso): {type(model2)}")
print(f"Model type (MLP): {type(model3)}")
print(f"Model type (Stacking): {type(model4)}")

# Load the real_estate data (if needed)
real_estate_data = pd.read_csv('Real estate price.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      try:
     
        age = float(request.form['age'])
        distance = float(request.form['distance'])
        store = float(request.form['store'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        # In ra các giá trị đầu vào
       print(f"Age: {age}, Distance: {distance}, Store: {store}, Latitude: {latitude}, Longitude: {longitude}")

        # Create feature array for prediction
        features = np.array([[age, distance, store, latitude, longitude]])

        # Predict using the loaded model
          scaled_features = scaler.transform(features)

            # Dự đoán bằng các mô hình
        prediction_linear = model.predict(scaled_features)[0]
        prediction_lasso = model2.predict(scaled_features)[0]
        prediction_mlp = model3.predict(scaled_features)[0]
        prediction_stacking = model4.predict(scaled_features)[0]

        # Trả về kết quả dự đoán trên giao diện web
        return render_template('index.html',
                               prediction_linear=prediction_linear,
                               prediction_lasso=prediction_lasso,
                               prediction_mlp=prediction_mlp,
                               prediction_stacking=prediction_stacking)

        except Exception as e:
            return f"Đã xảy ra lỗi: {e}"

if __name__ == '__main__':
    app.run(debug=True)
