import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Đọc dữ liệu
df = pd.read_csv('housing.csv')

# Chia dữ liệu thành X và y
X = df.drop(' Y house price of unit area', axis=1)
y = df['y house price of unit areae']

# Xác định các cột số và cột phân loại
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['Y house price of unit area']

# Tạo bộ tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm tạo pipeline
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Huấn luyện mô hình Linear Regression
lr_pipeline = create_pipeline(LinearRegression())
lr_pipeline.fit(X_train, y_train)

# Huấn luyện mô hình lasso Regression
lasso_pipeline = create_pipeline(Lasso(alpha=1.0))
lasso_pipeline.fit(X_train, y_train)

# Huấn luyện mô hình Neural Network (MLPRegressor)
mlp_pipeline = create_pipeline(MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
mlp_pipeline.fit(X_train, y_train)

# Huấn luyện mô hình Stacking Regressor
estimators = [
    ('lr', LinearRegression()),
    ('lasso', Lasso(alpha=1.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
]

stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', StackingRegressor(
        estimators=estimators,
        final_estimator=Lasso()
    ))
])
stacking_pipeline.fit(X_train, y_train)

# Tạo thư mục 'models' nếu chưa tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# Lưu các mô hình
joblib.dump(lr_pipeline, 'models/linear.joblib')
joblib.dump(lasso_pipeline, 'models/lasso.joblib')
joblib.dump(mlp_pipeline, 'models/mlp.joblib')
joblib.dump(stacking_pipeline, 'models/stacking.joblib')

print("Huấn luyện và lưu mô hình thành công!")