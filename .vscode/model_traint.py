import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib 

# Đọc dữ liệu từ file CSV
df = pd.read_csv('bmi.csv')
X = df[["Chieu cao", "Can nang"]]
y = df['chi so BMI']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình 
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lasso_model(X_train, y_train):
    model = Lasso(alpha=10.0)
    model.fit(X_train, y_train)
    return model

def train_neural_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

#
linear_model = train_linear_model(X_train, y_train)
neural_model = train_neural_model(X_train, y_train)
lasso_model = train_lasso_model(X_train, y_train)

# Lưu 
joblib.dump(linear_model, 'linear.joblib')
joblib.dump(neural_model, 'mlp.joblib')
joblib.dump(lasso_model, 'lasso.joblib')

#
y_pred_linear = linear_model.predict(X_test)
y_pred_neural = neural_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)

# Hàm đánh giá
def evaluate_model(y_test, y_pred, model_name):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nse = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    print(f"--- {model_name} ---")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NSE: {nse:.4f}")
    print()

# 
evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_neural, "Neural Network")
evaluate_model(y_test, y_pred_lasso, "lasso Regression")

# Stacking
base_models = [
    ('linear', linear_model),
    ('neural', neural_model),
    ('lasso', lasso_model)
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

stacking_model.fit(X_train, y_train)

y_pred_stacking = stacking_model.predict(X_test)

# Đánh giá mô hình stacking
evaluate_model(y_test, y_pred_stacking, "Stacking")

# Lưu mô hình stacking
joblib.dump(stacking_model, 'stacking.joblib')

# Hàm đánh giá cross-validation
def evaluate_model_cv(model, X, y, model_name):
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"--- {model_name} (Cross-validation R^2 scores) ---")
    print(scores)
    print(f"Mean R^2: {np.mean(scores):.4f}\n")

# Đánh giá cross-validation cho các mô hình
evaluate_model_cv(linear_model, X, y, "Linear Regression")
evaluate_model_cv(neural_model, X, y, "Neural Network")
evaluate_model_cv(lasso_model, X, y, "lasso Regression")
evaluate_model_cv(stacking_model, X, y, "Stacking")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Linear Regression', s=30)
plt.scatter(y_test, y_pred_stacking, color='green', label='Stacking', s=30)
plt.scatter(y_test, y_pred_lasso, color='purple', label='lasso', s=30)
plt.scatter(y_test, y_pred_neural, color='orange', label='Neural', s=30)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Đường hoàn hảo')
plt.title('So sánh Linear Regression và Stacking')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.legend()
plt.grid(True)
plt.show()