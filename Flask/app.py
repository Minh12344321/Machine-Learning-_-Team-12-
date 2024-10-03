from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Tải các mô hình
model_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model = joblib.load(os.path.join(model_dir, 'linear.joblib'))
lasso_model = joblib.load(os.path.join(model_dir, 'lasso.joblib'))
mlp_model = joblib.load(os.path.join(model_dir, 'mlp.joblib'))
stacking_model = joblib.load(os.path.join(model_dir, 'stacking.joblib'))

# Tạo dictionary cho các mô hình
models = {
    'Linear Regression': lr_model,
    'Lasso': lasso_model,
    'Neural Network': mlp_model,
    'Stacking Regressor': stacking_model
}

# Trang chủ với form nhập liệu
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        data = request.form.to_dict()
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame([data])
       
        # Kiểm tra giá trị thiếu
        if df.isnull().any().any():
            return "Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra lại."
        
        # Lấy tên mô hình được chọn
        model_name = request.form.get('model')
        
        # Lấy mô hình tương ứng
        model = models.get(model_name)
        
        # Dự đoán kết quả
        try:
            prediction = model.predict(df)[0]
        except Exception as e:
            return f"Lỗi trong quá trình dự đoán: {e}"
        
        # Lấy thông tin độ tin cậy (ví dụ: RMSE trên tập kiểm tra)
        # Đây là giá trị giả định, bạn nên tính RMSE thực tế từ quá trình huấn luyện
        model_scores = {
            'Linear Regression': 69043.17,
            'Lasso Regression': 69043.17,
            'Neural Network': 56023.45,
            'Stacking Regressor': 55012.34
        }
        confidence = model_scores.get(model_name)
        
        return render_template('result.html', prediction=prediction, confidence=confidence, model_name=model_name)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)