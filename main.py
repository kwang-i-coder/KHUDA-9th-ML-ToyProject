import pandas as pd
import numpy as np
import os
import warnings

from eda import plot_feature_importances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
import optuna

from rename import rename_map
from evaluate import get_result
from eda import plot_feature_importances


warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)


# 데이터 로딩
df = pd.read_csv(os.path.join(BASE_DIR, 'ultradata_all_recomputed_v2.csv'), encoding='utf-8-sig')
print(f"로드 완료. shape: {df.shape}")

# 칼럼명 rename에 따라서 변경
df = df.rename(columns=rename_map)
print(f"rename 완료. 컬럼 수: {len(df.columns)}")

## 결측치 제거
df = df[df['cluster_id'].notna()]
print(f"Cluster 결측 제거 후: {df.shape}")

### 데이터 전처리 작업

# 범주형 매핑 (한국말 시각화했을때 깨져서)
house_map = {
    '다가구'   : 'Multi_Family',
    '다세대'   : 'Multi_Unit',
    '단독'     : 'Single_House',
    '연립'     : 'Row_House',
    '연립다세대': 'Villa',
    '오피스텔' : 'Officetel',
}
ori_map = {
    '남향'   : 'South',
    '동향'   : 'East',
    '서향'   : 'West',
    '북향'   : 'North',
    '남동향' : 'South_East',
    '남서향' : 'South_West',
    'Unknown': 'Unknown',
    'unknown': 'Unknown',
}

#결측치 처리
df['House_Type']  = df['House_Type'].map(house_map).fillna('Other_House')
df['Orientation'] = df['Orientation'].map(ori_map).fillna('Other_Ori')

# 수치형 변환 (,제거 및 타입 변환)
df['Deposit']      = pd.to_numeric(df['Deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Monthly_Rent'] = pd.to_numeric(df['Monthly_Rent'], errors='coerce').fillna(0)

# 전월세 구분
df['Rent_Type'] = df['Monthly_Rent'].apply(lambda x: 'Monthly' if x > 0 else 'Jeonse')

# 환산가 계산
df['Price_Hwansan'] = df['Deposit'] + (df['Monthly_Rent'] * 12) / 0.6
if 'Price_Hwansan_Raw' in df.columns:
    raw = pd.to_numeric(df['Price_Hwansan_Raw'], errors='coerce')
    df['Price_Hwansan'] = raw.fillna(df['Price_Hwansan'])

## 타입 변환 및 결측치 처리 (중앙값으로)
df['Build_Year']       = pd.to_numeric(df['Build_Year'], errors='coerce')
df['House_Age']        = 2026 - df['Build_Year'].fillna(df['Build_Year'].median())
df['Area_m2']          = pd.to_numeric(df['Area_m2'], errors='coerce').fillna(df['Area_m2'].median())
df['Floor']            = pd.to_numeric(df['Floor'], errors='coerce')
df['School_Walk_Min']  = pd.to_numeric(df['School_Walk_Min'], errors='coerce')
df['School_Walk_Dist'] = pd.to_numeric(df['School_Walk_Dist'], errors='coerce')
df['Transit_Min']      = pd.to_numeric(df['Transit_Min'], errors='coerce')

# CSV에서 문자열로 읽힌 나머지 수치형 컬럼 일괄 변환
# (House_Type, Orientation 등 범주형만 제외, 나머지 object는 숫자로 강제 변환)
_categorical_cols = {'House_Type', 'Orientation', 'Road_Name', 'Cluster', 'Cluster_Clean',
                     'Cluster_Name', '대표주소', 'cluster_id', 'Rent_Type'}
for _col in df.select_dtypes(include='object').columns:
    if _col not in _categorical_cols:
        df[_col] = pd.to_numeric(df[_col], errors='coerce')

## 100 보다 작은 값은 노이즈로 제거
df = df[df['Price_Hwansan'] > 100]
print(f"이상치 제거 후: {df.shape}")

# 상하위 3% 제거 (전월세환산가 기준)
lower = df['Price_Hwansan'].quantile(0.03)
upper = df['Price_Hwansan'].quantile(0.97)
print(f"하위 3% 기준: {lower:.0f} 만원 / 상위 3% 기준: {upper:.0f} 만원")
df = df[(df['Price_Hwansan'] >= lower) & (df['Price_Hwansan'] <= upper)]
print(f"상하위 3% 제거 후: {df.shape}")

# 층(Floor) 결측 처리
print(f"\n층 결측 수 (처리 전): {df['Floor'].isna().sum()}")
df['Floor_Missing'] = df['Floor'].isna().astype(int)
df['Floor'] = df.groupby(['House_Type', 'cluster_id'])['Floor'].transform(
    lambda x: x.fillna(x.median())
)
df['Floor'] = df['Floor'].fillna(df['Floor'].median())
print(f"층 결측 수 (처리 후): {df['Floor'].isna().sum()}")

# 파생 변수 생성 (vif, 상관관계 분석 근거로 파생변수 생성, 특성 더 잘 드러내기 위해서 price_per_m2(발))
# 

ss = StandardScaler()
ss.fit(df[['School_Walk_Min']])
df['School_Walk_Min_scaled']  = ss.transform(df[['School_Walk_Min']])
ss.fit(df[['School_Walk_Dist']])
df['School_Walk_Dist_scaled'] = ss.transform(df[['School_Walk_Dist']])
df['Life_Infra_Score']  = df[['Conv_Count', 'Bank_Count', 'Hosp_Count', 'Cafe_Count', 'Rest_Count']].sum(axis=1)
df['Safety_Score']      = df['CCTV_500m'].fillna(0) + df['Lamp_500m'].fillna(0)
df['Transit_Score']     = df['Subway_Count'].fillna(0) + df['Bus_Count'].fillna(0) + df['Bus_200m'].fillna(0)
df['Pharm_Parking_Sum'] = df['Pharm_Count'].fillna(0) + df['Parking_Count'].fillna(0)
df['School_Access']     = (df['School_Walk_Min_scaled'].fillna(df['School_Walk_Min_scaled'].median()) +
                           df['School_Walk_Dist_scaled'].fillna(df['School_Walk_Dist_scaled'].median()) / 100)
df['Is_Low_Floor']  = (df['Floor'] <= 2).astype(int)
df['Is_High_Floor'] = (df['Floor'] >= 6).astype(int)
print("파생 변수 생성 완료")

# 종속 피처들 제거
drop_list = [
    'CCTV_300m', 'CCTV_1000m', 'Lamp_300m', 'Lamp_1000m','CCTV_500m', 'Lamp_500m',
    'Bus_100m', 'Bus_200m','Bus_Count',
    'Conv_Count', 'Bank_Count', 'Hosp_Count', 'Cafe_Count', 'Rest_Count', 'Mart_Count', 'School_Count_Near',
    'Pharm_Count', 'Parking_Count', 'Subway_Count', 'area_r'
] + [c for c in df.columns if '_Dist' in c or 'Dist' in c]

## 위도, 경도, 보증금, 월세 그리고 변경된 데이터 셋 반영한 피처 드롭 리스트
drop_list += [
    'Transit_Min', 'Build_Year', 'School_Walk_Dist', 'School_Walk_Min',
    'Road_Name', 'Lat', 'Lon', 'Cluster', '대표주소', 'center_lat', 'center_lon',
    'Deposit', 'Monthly_Rent', 'Price_Hwansan_Raw', 'lat_r', 'lon_r', 'center_lat', 'center_lon', 'area_r'
]
## drop_list에 있는 칼럼 제거
df_model = df.drop(columns=[c for c in drop_list if c in df.columns]).copy()
print(f"\n피처 선택 후 shape: {df_model.shape}")
## 어떻게 결측치를 채울지
count_score_cols = [c for c in df_model.columns
                    if any(k in c for k in ['Count', 'Score', 'Sum', 'Transit', 'Bus',
                                            'Floor_Missing', 'Is_Low_Floor', 'Is_High_Floor'])]
numeric_cols_all = df_model.select_dtypes(include=[np.number]).columns.tolist()
other_num_cols   = [c for c in numeric_cols_all
                    if c not in count_score_cols and c != 'Price_Hwansan']
## 결측치 채우기 (0 또는 중앙값)
df_model[count_score_cols] = df_model[count_score_cols].fillna(0)
df_model[other_num_cols]   = df_model[other_num_cols].fillna(df_model[other_num_cols].median())

df_final = pd.get_dummies(df_model, columns=['House_Type', 'Rent_Type', 'Orientation'], drop_first=False, dtype=int)
baseline_cols = ['House_Type_Other_House', 'Rent_Type_Monthly', 'Orientation_Unknown']
df_final = df_final.drop(columns=[c for c in baseline_cols if c in df_final.columns])
## cluster_id 제거
df_final = df_final.drop(columns=['cluster_id'], errors='ignore')
print(f"\n최종 피처 수: {df_final.shape[1] - 1}")
print(f"\n피처 목록: {df_final.columns.tolist()}")

## 학습 테스트 분리

X      = df_final.drop(columns=['Price_Hwansan'])
y_orig = df_final['Price_Hwansan'] ## 타겟값 설정
y_log  = np.log1p(y_orig) ## 로그 스케일 변환

X_train, X_test, y_log_train, y_log_test, y_orig_train, y_orig_test = train_test_split(
    X, y_log, y_orig, test_size=0.2, random_state=42
)
print(f"\n학습: {X_train.shape}, 테스트: {X_test.shape}")

all_results        = []
feature_importances = {}

## 모델 학습 및 평가

print("\n--- RandomForest ---")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_log_train)
y_pred_rf = np.expm1(rf.predict(X_test))
all_results.append(get_result('RandomForest (Base)', y_orig_test, y_pred_rf, y_orig_train))
feature_importances['RandomForest'] = pd.Series(rf.feature_importances_, index=X.columns)

print("\n--- XGBoost ---")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_log_train)
y_pred_xgb = np.expm1(xgb_model.predict(X_test))
all_results.append(get_result('XGBoost (Base)', y_orig_test, y_pred_xgb, y_orig_train))
feature_importances['XGBoost'] = pd.Series(xgb_model.feature_importances_, index=X.columns)

print("\n--- LightGBM ---")
lgb_model = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train, y_log_train)
y_pred_lgb = np.expm1(lgb_model.predict(X_test))
all_results.append(get_result('LightGBM (Base)', y_orig_test, y_pred_lgb, y_orig_train))
feature_importances['LightGBM'] = pd.Series(lgb_model.feature_importances_, index=X.columns)


print("\n--- Optuna 하이퍼파라미터 최적화 ---")
N_TRIALS = 50

print("\n[XGBoost Optuna - 50 trials]")
def xgb_objective(trial):
    params = {
        'objective'        : 'reg:squarederror',
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'gamma'            : trial.suggest_float('gamma', 0, 1),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'random_state': 42, 'n_jobs': -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_log_train, eval_set=[(X_test, y_log_test)], verbose=0)
    pred = np.expm1(model.predict(X_test))
    return np.sqrt(mean_squared_error(y_orig_test, pred))

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"XGBoost Best RMSE: {xgb_study.best_value:.2f} 만원")
print(f"Best params: {xgb_study.best_params}")

xgb_opt = xgb.XGBRegressor(**xgb_study.best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_opt.fit(X_train, y_log_train)
y_pred_xgb_opt = np.expm1(xgb_opt.predict(X_test))
all_results.append(get_result('XGBoost (Optuna)', y_orig_test, y_pred_xgb_opt, y_orig_train))
feature_importances['XGBoost (Optuna)'] = pd.Series(xgb_opt.feature_importances_, index=X.columns)

print("\n[LightGBM Optuna - 50 trials]")
def lgb_objective(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'        : trial.suggest_int('max_depth', 3, 12),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 150),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_log_train)
    pred = np.expm1(model.predict(X_test))
    return np.sqrt(mean_squared_error(y_orig_test, pred))

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"LightGBM Best RMSE: {lgb_study.best_value:.2f} 만원")
print(f"Best params: {lgb_study.best_params}")

lgb_opt = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, n_jobs=-1, verbose=-1)
lgb_opt.fit(X_train, y_log_train)
y_pred_lgb_opt = np.expm1(lgb_opt.predict(X_test))
all_results.append(get_result('LightGBM (Optuna)', y_orig_test, y_pred_lgb_opt, y_orig_train))
feature_importances['LightGBM (Optuna)'] = pd.Series(lgb_opt.feature_importances_, index=X.columns)

print("\n[RandomForest Optuna - 50 trials]")
def rf_objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 100, 400),
        'max_depth'       : trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features'    : trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42, 'n_jobs': -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_log_train)
    pred = np.expm1(model.predict(X_test))
    return np.sqrt(mean_squared_error(y_orig_test, pred))

rf_study = optuna.create_study(direction='minimize')
rf_study.optimize(rf_objective, n_trials=30, show_progress_bar=False)
print(f"RandomForest Best RMSE: {rf_study.best_value:.2f} 만원")
print(f"Best params: {rf_study.best_params}")

rf_opt = RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)
rf_opt.fit(X_train, y_log_train)
y_pred_rf_opt = np.expm1(rf_opt.predict(X_test))
all_results.append(get_result('RandomForest (Optuna)', y_orig_test, y_pred_rf_opt, y_orig_train))
feature_importances['RandomForest (Optuna)'] = pd.Series(rf_opt.feature_importances_, index=X.columns)

print("\n--- Stacking 앙상블 ---")

base_models = [xgb_opt, lgb_opt, rf_opt]
base_names  = ['XGBoost', 'LightGBM', 'RandomForest']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# OOF(Out-of-Fold) 예측 행렬 준비
oof_train = np.zeros((len(X_train), len(base_models)))
test_preds = np.zeros((len(X_test),  len(base_models)))

for j, (model, name) in enumerate(zip(base_models, base_names)):
    fold_test = np.zeros((len(X_test), kf.n_splits))
    for i, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr          = y_log_train.iloc[tr_idx]
        m = clone(model)
        m.fit(X_tr, y_tr)
        oof_train[val_idx, j] = m.predict(X_val)
        fold_test[:, i]       = m.predict(X_test)
    test_preds[:, j] = fold_test.mean(axis=1)
    print(f"  {name} OOF 완료")

# 메타 모델: Ridge 회귀
meta = Ridge(alpha=1.0)
meta.fit(oof_train, y_log_train)
y_pred_stack = np.expm1(meta.predict(test_preds))
all_results.append(get_result('Stacking (XGB+LGB+RF)', y_orig_test, y_pred_stack, y_orig_train))


#결과

results_df = pd.DataFrame(all_results)
best_idx   = results_df['RMSE'].idxmin()
best_name  = results_df.loc[best_idx, 'Model']

results_df.to_csv(os.path.join(BASE_DIR, 'model_results.csv'), index=False)

print("\n" + "=" * 60)
print("모든 분석 완료!")
print("=" * 60)
print(f"\n최고 성능 모델: {best_name}")
print(f"  R²  : {results_df.loc[best_idx, 'R2']:.4f}")
print(f"  RMSE: {results_df.loc[best_idx, 'RMSE']:.2f} 만원")
print(f"  MAPE: {results_df.loc[best_idx, 'MAPE']:.2f}%")
print(f"\n저장된 그래프: {SAVE_DIR}")
