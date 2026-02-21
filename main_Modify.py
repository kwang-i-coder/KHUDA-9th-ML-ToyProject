import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from rename import rename_map


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
    df = pd.read_csv('with_orientation_final.csv', encoding='utf-8')
except Exception:
    df = pd.read_csv('with_orientation_final.csv', encoding='cp949', errors='replace')


df = df.rename(columns=rename_map)
print("로드 완료:", df.shape)

# ── 2. 이상치 제거 (Cluster_Label == -1) ──────────────────────────────────────
df = df[df['Cluster_Label'] != -1]
print("Cluster -1 제거 후:", df.shape)

# ── 3. 범주형 값 영문 변환 ────────────────────────────────────────────────────
rent_map  = {'월세': 'Monthly', '전세': 'Jeonse'}
house_map = {'다가구': 'Multi_Family', '다세대': 'Multi_Unit',
             '단독': 'Single_House', '연립': 'Row_House', '연립다세대': 'Villa'}
ori_map   = {'남향': 'South', '동향': 'East', '서향': 'West', '북향': 'North',
             '남동향': 'South_East', '남서향': 'South_West', 'Unknown': 'Unknown'}

df['Rent_Type']   = df['Rent_Type'].map(rent_map).fillna('Other_Rent')
df['House_Type']  = df['House_Type'].map(house_map).fillna('Other_House')
df['Orientation'] = df['Orientation'].map(ori_map).fillna('Other_Ori')

# ── 4. 수치형 전처리 및 파생 변수 생성 ────────────────────────────────────────
df['Deposit']      = pd.to_numeric(df['Deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Monthly_Rent'] = pd.to_numeric(df['Monthly_Rent'], errors='coerce').fillna(0)

# 환산가 직접 계산 (보증금 + 월세*100)
df['Price_Hwansan'] = df['Deposit'] + (df['Monthly_Rent'] * 100)

# [추가] 이상치 제거 2 ────────────────────────────────────────
p_low = df['Price_Hwansan'].quantile(0.03)
p_high = df['Price_Hwansan'].quantile(0.97)
df = df[(df['Price_Hwansan'] >= p_low) & (df['Price_Hwansan'] <= p_high)]
#──────────────────────────────────────────────────────────────

# 건물 나이
df['Build_Year'] = pd.to_numeric(df['Build_Year'], errors='coerce')
df['House_Age']  = 2026 - df['Build_Year'].fillna(df['Build_Year'].median())

# 비정상 데이터 필터링
df = df[df['Price_Hwansan'] > 100]

# 파생 변수
df['Life_Infra_Score'] = df[['Conv_Count', 'Bank_Count', 'Hosp_Count',
                               'Cafe_Count', 'Rest_Count']].sum(axis=1)
df['Pharm_Parking_Sum'] = df['Pharm_Count'].fillna(0) + df['Parking_Count'].fillna(0)


print(f"\n--- 데이터 분리 완료 ---")
print(f"선택된 데이터 수: {len(df)}")
# ==========================================================

# ── 5. 불필요 컬럼 제거 ───────────────────────────────────────────────────────
drop_list = [
    'Transit_Min', 'Pharm_Count', 'Parking_Count', 'Build_Year', 'School_Walk_Dist',
    'Road_Name', 'Lat', 'Lon', 'Cluster_Label', 'Cluster', 'Cluster_Clean', 'Cluster_Name',
    'Deposit', 'Monthly_Rent', 'Conv_Count', 'Bank_Count', 'Hosp_Count', 'Cafe_Count', 'Rest_Count',
    'Price_Hwansan_Raw',  # 원본 환산값 제거 (직접 계산한 값 사용)
]
drop_list += [c for c in df.columns if '_Dist' in c]
df_model = df.drop(columns=drop_list, errors='ignore')

# ── 6. fillna 전략 분리 ───────────────────────────────────────────────────────
count_score_cols  = [c for c in df_model.columns if any(k in c for k in ['Count', 'Score', 'Sum'])]
numeric_cols      = df_model.select_dtypes(include=[np.number]).columns.tolist()
other_num_cols    = [c for c in numeric_cols if c not in count_score_cols and c != 'Price_Hwansan']

df_model[count_score_cols] = df_model[count_score_cols].fillna(0)
df_model[other_num_cols]   = df_model[other_num_cols].fillna(df_model[other_num_cols].median())

# ── 7. 원-핫 인코딩 (기준 컬럼 명시적 제거) ──────────────────────────────────
df_final = pd.get_dummies(df_model, columns=['House_Type', 'Rent_Type', 'Orientation'], drop_first=True)

baseline_cols = ['House_Type_Other_House', 'Rent_Type_Monthly', 'Orientation_Unknown']
df_final = df_final.drop(columns=[c for c in baseline_cols if c in df_final.columns])

print("\n최종 피처 수:", df_final.shape[1] - 1)  # Price_Hwansan 제외
print("Orientation 관련 피처:", [c for c in df_final.columns if 'Orientation' in c])

# ── 8. 학습/테스트 분리 ───────────────────────────────────────────────────────

X = df_final.drop(columns=['Price_Hwansan', 'Deposit', 'Monthly_Rent', 'Transit_Min'], errors='ignore')
y_orig = df_final['Price_Hwansan']
y_log  = np.log1p(y_orig)

X_train, X_test, y_log_train, y_log_test, y_orig_train, y_orig_test = train_test_split(
    X, y_log, y_orig, test_size=0.2, random_state=42
)

# ── 9. RandomForestRegressor ───────────────────────────────────────────────────
print("\n--- RandomForestRegressor 모델 학습 및 평가 ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_log_train)

y_pred_rf = np.expm1(rf_model.predict(X_test))
get_result(y_orig_test, y_pred_rf, y_orig_train)

importance_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[RandomForest Top 10 Feature Importances]")
print(importance_rf.head(10))

# ── 10. XGBoostRegressor ──────────────────────────────────────────────────────
print("\n--- XGBoostRegressor 모델 학습 및 평가 ---")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_log_train)

y_pred_xgb = np.expm1(xgb_model.predict(X_test))
get_result(y_orig_test, y_pred_xgb, y_orig_train)

importance_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[XGBoost Top 10 Feature Importances]")
print(importance_xgb.head(10))

print("\n최종 사용된 피처 리스트:")
print(X.columns.tolist())

import optuna
from sklearn.metrics import r2_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        # [수정] 6에서 9로 약간 풀어줍니다.
        'max_depth': trial.suggest_int('max_depth', 4, 9), 
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        # [수정] 제한을 살짝 완화합니다.
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10) 
    }
    
    model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_log_train)
    
    preds = np.expm1(model.predict(X_test))
    return r2_score(y_orig_test, preds)

print("\n--- Optuna XGBoost 튜닝 시작 ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # 30번 탐색

# ── 11. 최적 모델로 최종 평가 수행 ──────────────────────────────────────────

# Optuna가 찾아낸 최적의 파라미터 적용
best_params = study.best_params
final_xgb_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
final_xgb_model.fit(X_train, y_log_train)

# 예측 및 원복 (log -> exp)
y_pred_final_log = final_xgb_model.predict(X_test)
y_pred_final_orig = np.expm1(y_pred_final_log)

print(f"\n--- [최종] Optuna 최적화 XGBoost 모델 평가 ---")
# 기존에 정의된 get_result 함수 호출
get_result(y_orig_test, y_pred_final_orig, y_orig_train)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. 최적 파라미터로 최종 모델 재학습 및 예측
best_xgb = xgb.XGBRegressor(**study.best_params, objective='reg:squarederror', random_state=42)
best_xgb.fit(X_train, y_log_train)

y_pred_log = best_xgb.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)
y_actual_orig = np.expm1(y_log_test)

# 2. 오차(Residual) 계산 및 데이터프레임 생성
results_df = pd.DataFrame({
    'Actual': y_actual_orig,
    'Predicted': y_pred_orig,
    'Error': y_actual_orig - y_pred_orig,
    'Abs_Error_Pct': np.abs((y_actual_orig - y_pred_orig) / y_actual_orig) * 100
})

# 3. 시각화: 실제값 vs 예측값 산점도
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_actual_orig, y=y_pred_orig, alpha=0.5)
plt.plot([y_actual_orig.min(), y_actual_orig.max()], [y_actual_orig.min(), y_actual_orig.max()], 'r--', lw=2)
plt.title('Actual vs Predicted (Price_Hwansan)')
plt.xlabel('Actual Price (10k KRW)')
plt.ylabel('Predicted Price (10k KRW)')

# 4. 시각화: 오차 분포 (Residual Plot)
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_orig, y=results_df['Error'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Residual Plot (Errors by Price Level)')
plt.xlabel('Predicted Price')
plt.ylabel('Error (Actual - Predicted)')

plt.tight_layout()
plt.show()

# 5. 오차가 큰 상위 15% 데이터 분석
top_15_err = results_df.nlargest(int(len(results_df) * 0.15), 'Abs_Error_Pct')
print(f"\n--- 오차 상위 15% 데이터 요약 (오차율 기준) ---")
print(top_15_err.describe())

# 실제보다 훨씬 비싼데 낮게 예측된 '저평가 오류' 상위 5개 출력
print("\n[Underestimated: 실제보다 훨씬 낮게 예측된 사례]")
print(top_15_err[top_15_err['Error'] > 0].sort_values(by='Error', ascending=False).head(5))

import matplotlib.pyplot as plt
import seaborn as sns

# 한 화면에 두 모델의 중요도를 시각화하는 함수
def plot_combined_importance(rf_model, xgb_model, features):
    # 데이터 준비
    rf_imp = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False).head(15)
    xgb_imp = pd.DataFrame({'Feature': features, 'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=False).head(15)

    # 시각화 설정 (1행 2열)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # RandomForest 그래프 (왼쪽)
    sns.barplot(x='Importance', y='Feature', data=rf_imp, ax=axes[0], palette='Blues_d')
    axes[0].set_title('RandomForest Top 15 Features', fontsize=14)
    axes[0].set_xlabel('Importance Score')

    # XGBoost 그래프 (오른쪽)
    sns.barplot(x='Importance', y='Feature', data=xgb_imp, ax=axes[1], palette='Oranges_d')
    axes[1].set_title('XGBoost (Optimized) Top 15 Features', fontsize=14)
    axes[1].set_xlabel('Importance Score')

    plt.suptitle('Model Feature Importance Comparison', fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()

    # 수치 비교를 위한 테이블 출력
    comparison_df = pd.DataFrame({
        'Rank': range(1, 11),
        'RandomForest': rf_imp['Feature'].values[:10],
        'XGBoost': xgb_imp['Feature'].values[:10]
    })
    print("\n[상위 10개 피처 비교 요약]")
    print(comparison_df.to_string(index=False))

# 그래프 실행 (best_xgb는 Optuna로 학습된 모델)
plot_combined_importance(rf_model, best_xgb, X_train.columns)
