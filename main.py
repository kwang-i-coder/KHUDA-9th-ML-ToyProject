import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from rename import rename_map
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge


# ── 평가 함수 (y_orig_train을 인자로 명시적으로 받음) ──────────────────────────
def get_result(y_orig_test, y_pred_orig, y_orig_train):
    r2   = r2_score(y_orig_test, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred_orig))
    mape = mean_absolute_percentage_error(y_orig_test, y_pred_orig)
    mae  = mean_absolute_error(y_orig_test, y_pred_orig)
    mae_naive = mean_absolute_error(
        y_orig_test,
        np.full(len(y_orig_test), y_orig_train.mean())   # dtype 안전 처리
    )
    mase = mae / mae_naive

    print("=== 로그 변환 후 원복 데이터(만원 단위) 평가 결과 ===")
    print(f"1. R-Squared (결정계수)          : {r2:.4f}")
    print(f"2. RMSE (평균 제곱근 오차)       : {rmse:.2f} 만원")
    print(f"3. MAPE (평균 절대 백분율 오차)  : {mape*100:.2f}%")
    print(f"4. MASE (평균 절대 척도 오차)    : {mase:.4f}")
    print(f"5. MAE  (평균 절대 오차)         : {mae:.2f} 만원")
    print("=" * 50)


# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
try:
    df = pd.read_csv('final_data.csv', encoding='utf-8')
except Exception:
    df = pd.read_csv('final_data.csv', encoding='cp949', errors='replace')

df = df.rename(columns=rename_map)
print("로드 완료:", df.shape)

# 추가적인 feature engineering(gpt)
df["Age_squared"] = df["House_Age"] ** 2
df["New_House"] = (df["House_Age"] <= 5).astype(int)
df["Old_House"] = (df["House_Age"] >= 20).astype(int)
df["Jeonse_Villa"] = df["Rent_Type_Jeonse"] * df["House_Type_Villa"]
df["Jeonse_Multi"] = df["Rent_Type_Jeonse"] * df["House_Type_Multi_Family"]
df["Infra_density"] = df["Life_Infra_Score"] / df["Area_m2"]

# ── 8. 학습/테스트 분리 ───────────────────────────────────────────────────────
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
# importance_stacking = pd.Series(stack_model.feature_importances_, index=X.columns).sort_values(ascending=False)
# print("\n[Stacking Top 10 Feature Importances]")
# print(importance_stacking.head(10))

print("\n최종 사용된 피처 리스트:")
print(X.columns.tolist())
