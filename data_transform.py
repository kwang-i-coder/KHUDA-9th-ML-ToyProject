import pandas as pd
import numpy as np
from main import get_result
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rename import rename_map
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

df = pd.read_csv('with_cctv_lamp_bus.csv', encoding='utf-8')
df = df.drop(columns=['Unnamed: 0', '도로명', 'lat', 'lon', 'cluster', 'cluster_name', 'cluster_label', '보증금(만원)', '월세금(만원)', 'cctv_300m', 'cctv_1000m', 'lamp_300m', 'lamp_1000m', 'nearby_bus_count(100m)', 'nearby_bus_count(200m)', 'subway_count', 'subway_min_distance', '대중교통시간(분)', 'bank_min_distance', 'hospital_min_distance', 'pharmacy_min_distance', 'cafe_min_distance', 'restaurant_min_distance', 'mart_min_distance', 'parking_min_distance'], errors='ignore')
df['Age'] = 2026 - df['건축년도']
df = df.drop(columns=['건축년도'])
df.rename(columns={'주택유형':'House_Type'}, inplace=True)
df.rename(columns={'향':'Orientation'}, inplace=True)

# main에 있던 형식으로 변환
# ── 3. 범주형 값 영문 변환 ────────────────────────────────────────────────────
house_map = {'다가구': 'Multi_Family', '다세대': 'Multi_Unit',
             '단독': 'Single_House', '연립': 'Row_House', '연립다세대': 'Villa'}
ori_map   = {'남향': 'South', '동향': 'East', '서향': 'West', '북향': 'North',
             '남동향': 'South_East', '남서향': 'South_West', 'Unknown': 'Unknown'}
df['House_Type']  = df['House_Type'].map(house_map).fillna('Other_House')
df['Orientation'] = df['Orientation'].map(ori_map).fillna('Other_Ori')

df = pd.get_dummies(df, columns=['House_Type', 'Orientation'], drop_first=False)

commercial_cols = [
    "cafe_count",
    "restaurant_count",
    "mart_count",
    "pharmacy_count",
    "convenience_store_count"
]

df["commercial_index"] = df[commercial_cols].sum(axis=1)
df = df.drop(columns=commercial_cols)

# 모든 시설들을 한데 모은 열을 만들고 cctv 데이터 놔둔 상태
# 학습

X      = df.drop(columns=['Price_Hwansan'])
y_orig = df['Price_Hwansan']
y_log  = np.log1p(y_orig)

X_train, X_test, y_log_train, y_log_test, y_orig_train, y_orig_test = train_test_split(
    X, y_log, y_orig, test_size=0.2, random_state=42
)

# ── 9. RandomForestRegressor ───────────────────────────────────────────────────
print("\n--- RandomForestRegressor 모델 학습 및 평가 ---")
param_dist = {
    'n_estimators': [300, 500, 800],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_log_train)
params = random_search.best_params_
rf_model = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=42, min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], max_depth=params['max_depth'])
rf_model.fit(X_train, y_log_train)

y_pred_rf = np.expm1(rf_model.predict(X_test))
get_result(y_orig_test, y_pred_rf, y_orig_train)

importance_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[RandomForest Top 10 Feature Importances]")
print(importance_rf.head(10))

# ── 10. XGBoostRegressor ──────────────────────────────────────────────────────
print("\n--- XGBoostRegressor 모델 학습 및 평가 ---")
param_dist = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [500, 800, 1000],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_log_train)
params = random_search.best_params_
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=params['n_estimators'], random_state=42, learning_rate=params['learning_rate'], max_depth=params['max_depth'], sumsample=params['subsample'], colsample_bytree=params['colsample_bytree'], min_child_weight=params['min_child_weight'])
xgb_model.fit(X_train, y_log_train)

y_pred_xgb = np.expm1(xgb_model.predict(X_test))
get_result(y_orig_test, y_pred_xgb, y_orig_train)

importance_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[XGBoost Top 10 Feature Importances]")
print(importance_xgb.head(10))

print("\n--- LightGBM 모델 학습 및 평가 ---")
X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
param_dist = {
    'learning_rate': [0.01, 0.03, 0.05],
    'num_leaves': [31, 64, 100, 150],
    'max_depth': [3, 5, 7, -1],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}
lgbm_wrapper = LGBMRegressor(n_estimators=400, learning_rate=0.05)
random_search = RandomizedSearchCV(
    lgbm_wrapper,
    param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
random_search.fit(X_train, y_log_train)
params = random_search.best_params_
lgbm_wrapper = LGBMRegressor(learning_rate=params['learning_rate'], num_leaves=params['num_leaves'], max_depth=params['max_depth'], min_child_samples=params['min_child_samples'], subsample=params['subsample'], colsample_bytree=params['colsample_bytree'], verbose=-1)
lgbm_wrapper.fit(X_train, y_log_train)
y_pred_lgbm = np.expm1(lgbm_wrapper.predict(X_test))
get_result(y_orig_test, y_pred_lgbm, y_orig_train)
importance_lgbm = pd.Series(lgbm_wrapper.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[LightGBM Top 10 Feature Importances]")
print(importance_lgbm.head(10))

print("\n--- Stacking Ensemble 모델 학습 및 평가 ---")
estimators = [
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgbm', lgbm_wrapper)
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)
stack_model.fit(X_train, y_log_train)
y_pred_stack = np.expm1(stack_model.predict(X_test))
get_result(y_orig_test, y_pred_stack, y_orig_train)
importance_stacking = pd.Series(stack_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[Stacking Top 10 Feature Importances]")
print(importance_stacking.head(10))

print("\n최종 사용된 피처 리스트:")
print(X.columns.tolist())
