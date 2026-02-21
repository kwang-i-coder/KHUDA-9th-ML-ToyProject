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

# 건물 나이
df['Build_Year'] = pd.to_numeric(df['Build_Year'], errors='coerce')
df['House_Age']  = 2026 - df['Build_Year'].fillna(df['Build_Year'].median())

# 비정상 데이터 필터링
df = df[df['Price_Hwansan'] > 100]

# 파생 변수
df['Life_Infra_Score'] = df[['Conv_Count', 'Bank_Count', 'Hosp_Count',
                               'Cafe_Count', 'Rest_Count']].sum(axis=1)
df['Pharm_Parking_Sum'] = df['Pharm_Count'].fillna(0) + df['Parking_Count'].fillna(0)

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
df_final = pd.get_dummies(df_model, columns=['House_Type', 'Rent_Type', 'Orientation'], drop_first=False)

baseline_cols = ['House_Type_Other_House', 'Rent_Type_Monthly', 'Orientation_Unknown']
df_final = df_final.drop(columns=[c for c in baseline_cols if c in df_final.columns])

print("\n최종 피처 수:", df_final.shape[1] - 1)  # Price_Hwansan 제외
print("Orientation 관련 피처:", [c for c in df_final.columns if 'Orientation' in c])

# ── 8. 학습/테스트 분리 ───────────────────────────────────────────────────────
X      = df_final.drop(columns=['Price_Hwansan'])
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
