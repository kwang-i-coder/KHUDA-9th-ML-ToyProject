import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 폰트 설정 (Windows용) ────────────────────────────────────────────────────
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: 데이터 로드")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_DIR, 'ultradata_all_recomputed_v2.csv'), encoding='utf-8-sig')
print(f"로드 완료. shape: {df.shape}")
print(f"컬럼 수: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. rename_map 적용
# ══════════════════════════════════════════════════════════════════════════════
rename_map = {
    '전용면적(㎡)'         : 'Area_m2',
    '보증금(만원)'         : 'Deposit',
    '월세금(만원)'         : 'Monthly_Rent',
    '건축년도'             : 'Build_Year',
    '도로명'               : 'Road_Name',
    '주택유형'             : 'House_Type',
    'lat'                  : 'Lat',
    'lon'                  : 'Lon',
    '층'                   : 'Floor',
    '학교까지도보시간(분)' : 'School_Walk_Min',
    '도보거리(m)'          : 'School_Walk_Dist',
    '대중교통시간(분)'     : 'Transit_Min',
    'convenience_store_count'        : 'Conv_Count',
    'convenience_store_min_distance' : 'Conv_Dist',
    'subway_count'         : 'Subway_Count',
    'subway_min_distance'  : 'Subway_Dist',
    'bank_count'           : 'Bank_Count',
    'bank_min_distance'    : 'Bank_Dist',
    'hospital_count'       : 'Hosp_Count',
    'hospital_min_distance': 'Hosp_Dist',
    'pharmacy_count'       : 'Pharm_Count',
    'pharmacy_min_distance': 'Pharm_Dist',
    'school_count'         : 'School_Count_Near',
    'school_min_distance'  : 'School_Dist_Near',
    'cafe_count'           : 'Cafe_Count',
    'cafe_min_distance'    : 'Cafe_Dist',
    'restaurant_count'     : 'Rest_Count',
    'restaurant_min_distance' : 'Rest_Dist',
    'mart_count'           : 'Mart_Count',
    'mart_min_distance'    : 'Mart_Dist',
    'parking_count'        : 'Parking_Count',
    'parking_min_distance' : 'Parking_Dist',
    # cluster 관련: ultradata는 cluster_id 하나만 존재
    'cluster_id'           : 'Cluster_Label',
    '대표주소'             : 'Cluster_Name',   # 모델링에서 제거 예정
    'center_lat'           : 'Center_Lat',     # 모델링에서 제거 예정
    'center_lon'           : 'Center_Lon',     # 모델링에서 제거 예정
    '전월세환산값'         : 'Price_Hwansan_Raw',
    '향'                   : 'Orientation',
    'cctv_300m'            : 'CCTV_300m',
    'lamp_300m'            : 'Lamp_300m',
    'cctv_500m'            : 'CCTV_500m',
    'lamp_500m'            : 'Lamp_500m',
    'cctv_1000m'           : 'CCTV_1000m',
    'lamp_1000m'           : 'Lamp_1000m',
    'bus_count'            : 'Bus_Count',
    'nearby_bus_count(100m)' : 'Bus_100m',
    'nearby_bus_count(200m)' : 'Bus_200m',
    'lat_r'                : 'Lat_R',          # 모델링에서 제거 예정
    'lon_r'                : 'Lon_R',          # 모델링에서 제거 예정
    'area_r'               : 'Area_R',         # 모델링에서 제거 예정
}
df = df.rename(columns=rename_map)
print(f"rename 완료. 컬럼 수: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 기본 전처리
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: 기본 전처리")
print("=" * 60)

# cluster_id == -1 (노이즈) 제거
df = df[df['Cluster_Label'] != -1].reset_index(drop=True)
print(f"Cluster -1 제거 후: {df.shape}")

# 전체 컬럼의 'unknown' 문자열 → NaN 일괄 치환
df = df.replace('unknown', np.nan)

# object 타입으로 남은 수치형 컬럼 일괄 변환 (unknown 혼재로 string dtype 유지된 컬럼 처리)
_non_numeric = ['House_Type', 'Orientation', 'Road_Name', 'Cluster_Name']
for col in df.select_dtypes(include='object').columns:
    if col not in _non_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 주택유형 매핑 (오피스텔 추가)
house_map = {
    '다가구'   : 'Multi_Family',
    '다세대'   : 'Multi_Unit',
    '단독'     : 'Single_House',
    '연립'     : 'Row_House',
    '연립다세대': 'Villa',
    '오피스텔' : 'Officetel',
}

# 향 매핑 (unknown 소문자 통일)
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

df['House_Type']  = df['House_Type'].map(house_map).fillna('Other_House')
df['Orientation'] = df['Orientation'].map(ori_map).fillna('Other_Ori')

# 수치형 변환
df['Deposit']      = pd.to_numeric(df['Deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Monthly_Rent'] = pd.to_numeric(df['Monthly_Rent'], errors='coerce').fillna(0)

# 전월세 구분: 원본 컬럼 없으므로 월세금 > 0 여부로 구분
df['Rent_Type'] = df['Monthly_Rent'].apply(lambda x: 'Monthly' if x > 0 else 'Jeonse')

# 환산가 재계산 (보증금 + 월세금 × 100)
df['Price_Hwansan'] = df['Deposit'] + (df['Monthly_Rent'] * 100)
# Price_Hwansan_Raw 있으면 우선 사용 (결측은 재계산값으로 보완)
if 'Price_Hwansan_Raw' in df.columns:
    raw = pd.to_numeric(df['Price_Hwansan_Raw'], errors='coerce')
    df['Price_Hwansan'] = raw.fillna(df['Price_Hwansan'])

df['Build_Year']       = pd.to_numeric(df['Build_Year'], errors='coerce')
df['House_Age']        = 2026 - df['Build_Year'].fillna(df['Build_Year'].median())
df['Area_m2']          = pd.to_numeric(df['Area_m2'], errors='coerce').fillna(df['Area_m2'].median())
df['School_Walk_Min']  = pd.to_numeric(df['School_Walk_Min'], errors='coerce')
df['School_Walk_Dist'] = pd.to_numeric(df['School_Walk_Dist'], errors='coerce')
df['Transit_Min']      = pd.to_numeric(df['Transit_Min'], errors='coerce')

df = df[df['Price_Hwansan'] > 100]
print(f"이상치 제거 후: {df.shape}")
######### test
# 환산가 통계치 확인


# 상위 3% / 하위 3% 제거 (전월세환산가 기준)
lower = df['Price_Hwansan'].quantile(0.03)
upper = df['Price_Hwansan'].quantile(0.97)
print(f"하위 3% 기준: {lower:.0f} 만원 / 상위 3% 기준: {upper:.0f} 만원")
df = df[(df['Price_Hwansan'] >= lower) & (df['Price_Hwansan'] <= upper)]
print(f"상하위 3% 제거 후: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3-1. 층(Floor) 결측 처리
#   주택유형 + Cluster_Clean 그룹별 중위수로 대체
#   결측 여부 플래그 피처(Floor_Missing) 추가
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3-1: 층(Floor) 결측 처리")
print("=" * 60)

df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce')
print(f"층 결측 수 (처리 전): {df['Floor'].isna().sum()}")

# 플래그 먼저 생성 (원본 결측 위치 보존)
df['Floor_Missing'] = df['Floor'].isna().astype(int)

# 그룹별 중위수 대체
df['Floor'] = df.groupby(['House_Type', 'Cluster_Label'])['Floor'].transform(
    lambda x: x.fillna(x.median())
)
# 그룹 중위수로도 안 채워진 경우 → 전체 중위수
floor_global_median = df['Floor'].median()
df['Floor'] = df['Floor'].fillna(floor_global_median)

print(f"층 결측 수 (처리 후): {df['Floor'].isna().sum()}")
print(f"Floor_Missing=1 비율: {df['Floor_Missing'].mean():.2%}")
print(f"층 분포:\n{df['Floor'].describe()}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. EDA - 타겟 분포
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['Price_Hwansan'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('환산가 분포 (원본)', fontsize=14)
axes[0].set_xlabel('환산가 (만원)')
axes[0].set_ylabel('빈도')
axes[1].hist(np.log1p(df['Price_Hwansan']), bins=50, color='salmon', edgecolor='white')
axes[1].set_title('환산가 분포 (로그 변환)', fontsize=14)
axes[1].set_xlabel('log(1 + 환산가)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '01_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 01 saved: 타겟 분포")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 파생 변수 생성
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: 파생 변수 생성")
print("=" * 60)

df['Life_Infra_Score'] = df[['Conv_Count','Bank_Count','Hosp_Count','Cafe_Count','Rest_Count']].sum(axis=1)
df['Safety_Score']     = df['CCTV_500m'].fillna(0) + df['Lamp_500m'].fillna(0)
df['Transit_Score']    = df['Subway_Count'].fillna(0) + df['Bus_Count'].fillna(0) + df['Bus_200m'].fillna(0)
df['Price_per_m2']     = df['Price_Hwansan'] / (df['Area_m2'].clip(lower=1))
df['Pharm_Parking_Sum']= df['Pharm_Count'].fillna(0) + df['Parking_Count'].fillna(0)
df['School_Access']    = (df['School_Walk_Min'].fillna(df['School_Walk_Min'].median()) +
                          df['School_Walk_Dist'].fillna(df['School_Walk_Dist'].median()) / 100)

# 층 관련 파생: 저층(1-2층) / 고층(6층+) 여부
df['Is_Low_Floor']  = (df['Floor'] <= 2).astype(int)
df['Is_High_Floor'] = (df['Floor'] >= 6).astype(int)

print("파생 변수 생성 완료: Life_Infra_Score, Safety_Score, Transit_Score, "
      "Price_per_m2, Pharm_Parking_Sum, School_Access, Is_Low_Floor, Is_High_Floor")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 상관관계 분석
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: 상관관계 분석")
print("=" * 60)

numeric_cols_for_corr = [
    'Price_Hwansan', 'Area_m2', 'House_Age', 'Floor',
    'Conv_Count', 'Subway_Count', 'Bank_Count', 'Hosp_Count',
    'Pharm_Count', 'Cafe_Count', 'Rest_Count', 'Mart_Count', 'Parking_Count',
    'School_Walk_Min', 'Transit_Min',
    'CCTV_300m', 'CCTV_500m', 'CCTV_1000m',
    'Lamp_300m', 'Lamp_500m', 'Lamp_1000m',
    'Bus_Count', 'Bus_100m', 'Bus_200m',
    'Life_Infra_Score', 'Safety_Score', 'Transit_Score',
]
numeric_cols_for_corr = [c for c in numeric_cols_for_corr if c in df.columns]
corr_df = df[numeric_cols_for_corr].fillna(0).corr()

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 7})
ax.set_title('변수 간 상관관계 히트맵', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 02 saved: 상관관계 히트맵")

target_corr = corr_df['Price_Hwansan'].drop('Price_Hwansan').sort_values(key=abs, ascending=False)
print("\n환산가와 상관관계 TOP 10:")
print(target_corr.head(10))

fig, ax = plt.subplots(figsize=(10, 7))
top15 = target_corr.head(15).sort_values()
top15.plot(kind='barh', ax=ax, color=['salmon' if v < 0 else 'steelblue' for v in top15])
ax.set_title('환산가와 각 변수의 상관계수 (상위 15)', fontsize=14)
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '03_target_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 03 saved: 타겟 상관관계")

# ── 층 vs 환산가 시각화 추가 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df['Floor'], df['Price_Hwansan'], alpha=0.3, color='steelblue', s=15)
axes[0].set_title('층 vs 환산가', fontsize=13)
axes[0].set_xlabel('층')
axes[0].set_ylabel('환산가 (만원)')
floor_group = df.groupby('Floor')['Price_Hwansan'].median().reset_index()
axes[1].bar(floor_group['Floor'], floor_group['Price_Hwansan'], color='steelblue', alpha=0.8)
axes[1].set_title('층별 환산가 중위수', fontsize=13)
axes[1].set_xlabel('층')
axes[1].set_ylabel('환산가 중위수 (만원)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '03b_floor_vs_price.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 03b saved: 층 vs 환산가")

print("\nCCTV 스케일 간 상관관계:")
print(df[['CCTV_300m','CCTV_500m','CCTV_1000m']].fillna(0).corr())
print("\n가로등 스케일 간 상관관계:")
print(df[['Lamp_300m','Lamp_500m','Lamp_1000m']].fillna(0).corr())
print("\n버스 스케일 간 상관관계:")
print(df[['Bus_100m','Bus_200m','Bus_Count']].fillna(0).corr())

# ══════════════════════════════════════════════════════════════════════════════
# 7. VIF 분석
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: VIF (분산팽창계수) 분석")
print("=" * 60)

vif_cols = [
    'Area_m2', 'House_Age', 'Floor',
    'Conv_Count', 'Subway_Count', 'Bank_Count', 'Hosp_Count',
    'Pharm_Count', 'Cafe_Count', 'Rest_Count', 'Mart_Count', 'Parking_Count',
    'School_Walk_Min', 'Transit_Min',
    'CCTV_300m', 'CCTV_500m', 'CCTV_1000m',
    'Lamp_300m', 'Lamp_500m', 'Lamp_1000m',
    'Bus_Count', 'Bus_100m', 'Bus_200m',
    'Life_Infra_Score', 'Safety_Score', 'Transit_Score',
]
vif_cols = [c for c in vif_cols if c in df.columns]
vif_data = df[vif_cols].fillna(0).copy()
vif_data = vif_data.loc[:, vif_data.std() > 0]

scaler = StandardScaler()
vif_scaled = pd.DataFrame(scaler.fit_transform(vif_data), columns=vif_data.columns)

vif_results = []
for i, col in enumerate(vif_scaled.columns):
    try:
        vif_val = variance_inflation_factor(vif_scaled.values, i)
        vif_results.append({'Feature': col, 'VIF': vif_val})
    except:
        vif_results.append({'Feature': col, 'VIF': np.nan})

vif_df = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)
print("\nVIF 결과 (내림차순):")
print(vif_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red' if v > 10 else ('orange' if v > 5 else 'steelblue')
          for v in vif_df['VIF'].fillna(0)]
ax.barh(vif_df['Feature'], vif_df['VIF'].fillna(0), color=colors)
ax.axvline(5, color='orange', linestyle='--', label='VIF=5 (주의)')
ax.axvline(10, color='red', linestyle='--', label='VIF=10 (위험)')
ax.set_title('VIF (분산팽창계수) - 다중공선성 체크', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '04_vif_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 04 saved: VIF 분석")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 피처 선택
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: 피처 선택 결정")
print("=" * 60)

drop_list = [
    'CCTV_300m', 'CCTV_1000m', 'Lamp_300m', 'Lamp_1000m',
    'Bus_Count',
    #➡️상관계수 낮았던 bus 100m 는 삭제 하지 않음
    'Conv_Count', 'Bank_Count', 'Hosp_Count', 'Cafe_Count', 'Rest_Count',
    'Pharm_Count', 'Parking_Count',
    'Subway_Count',
] + [c for c in df.columns if '_Dist' in c or 'Dist' in c]

drop_list += [
    'Transit_Min', 'Build_Year', 'School_Walk_Dist',
    'Road_Name', 'Lat', 'Lon',
    'Cluster_Label', 'Cluster_Name', 'Center_Lat', 'Center_Lon',
    'Lat_R', 'Lon_R', 'Area_R',
    'Deposit', 'Monthly_Rent', 'Price_Hwansan_Raw', 'Price_per_m2', 'Floor_Missing',
]


df_model = df.drop(columns=[c for c in drop_list if c in df.columns])
print(f"피처 선택 후 shape: {df_model.shape}")
print("남은 수치형 컬럼:", [c for c in df_model.select_dtypes(include=[np.number]).columns
                            if c != 'Price_Hwansan'])

# ══════════════════════════════════════════════════════════════════════════════
# 9. fillna 및 인코딩
# ══════════════════════════════════════════════════════════════════════════════
count_score_cols = [c for c in df_model.columns
                    if any(k in c for k in ['Count','Score','Sum','Transit','Bus','Floor_Missing',
                                            'Is_Low_Floor','Is_High_Floor'])]
numeric_cols_all = df_model.select_dtypes(include=[np.number]).columns.tolist()
other_num_cols   = [c for c in numeric_cols_all
                    if c not in count_score_cols and c != 'Price_Hwansan']

df_model[count_score_cols] = df_model[count_score_cols].fillna(0)
df_model[other_num_cols]   = df_model[other_num_cols].fillna(df_model[other_num_cols].median())

df_final = pd.get_dummies(df_model, columns=['House_Type','Rent_Type','Orientation'], drop_first=False)
baseline_cols = ['House_Type_Other_House', 'Rent_Type_Monthly', 'Orientation_Unknown']
df_final = df_final.drop(columns=[c for c in baseline_cols if c in df_final.columns])

print(f"\n최종 피처 수: {df_final.shape[1] - 1}")
print("피처 목록:", [c for c in df_final.columns if c != 'Price_Hwansan'])

# ══════════════════════════════════════════════════════════════════════════════
# 10. 학습/테스트 분리
# ══════════════════════════════════════════════════════════════════════════════
X      = df_final.drop(columns=['Price_Hwansan'])
y_orig = df_final['Price_Hwansan']
y_log  = np.log1p(y_orig)

X_train, X_test, y_log_train, y_log_test, y_orig_train, y_orig_test = train_test_split(
    X, y_log, y_orig, test_size=0.2, random_state=42
)
print(f"\n학습: {X_train.shape}, 테스트: {X_test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. 평가 함수
# ══════════════════════════════════════════════════════════════════════════════
def get_result(model_name, y_orig_test, y_pred_orig, y_orig_train):
    r2   = r2_score(y_orig_test, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred_orig))
    mape = mean_absolute_percentage_error(y_orig_test, y_pred_orig)
    mae  = mean_absolute_error(y_orig_test, y_pred_orig)
    mae_naive = mean_absolute_error(y_orig_test, np.full(len(y_orig_test), y_orig_train.mean()))
    mase = mae / mae_naive
    results = {'Model': model_name, 'R2': r2, 'RMSE': rmse, 'MAPE': mape*100, 'MAE': mae, 'MASE': mase}
    print(f"\n=== {model_name} 평가 결과 ===")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.2f} 만원")
    print(f"  MAPE : {mape*100:.2f}%")
    print(f"  MAE  : {mae:.2f} 만원")
    print(f"  MASE : {mase:.4f}")
    return results

all_results = []
feature_importances = {}

# ══════════════════════════════════════════════════════════════════════════════
# 12. 베이스라인 모델들
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12: 베이스라인 모델 학습")
print("=" * 60)

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

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, (name, imp) in zip(axes, feature_importances.items()):
    imp_sorted = imp.sort_values(ascending=True).tail(15)
    imp_sorted.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'{name} 피처 중요도 (Top 15)', fontsize=12)
    ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '05_feature_importances_base.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 05 saved: 베이스라인 피처 중요도")

# ══════════════════════════════════════════════════════════════════════════════
# 13. Optuna 하이퍼파라미터 최적화
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 13: Optuna 하이퍼파라미터 최적화")
print("=" * 60)

N_TRIALS = 50

print("\n[XGBoost Optuna - 50 trials]")
def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'random_state': 42, 'n_jobs': -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_log_train, eval_set=[(X_test, y_log_test)], verbose=False)
    pred = np.expm1(model.predict(X_test))
    return mean_absolute_percentage_error(y_orig_test, pred) * 100

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"XGBoost Best MAPE: {xgb_study.best_value:.2f}%")
print(f"Best params: {xgb_study.best_params}")

xgb_opt = xgb.XGBRegressor(**xgb_study.best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_opt.fit(X_train, y_log_train)
y_pred_xgb_opt = np.expm1(xgb_opt.predict(X_test))
all_results.append(get_result('XGBoost (Optuna)', y_orig_test, y_pred_xgb_opt, y_orig_train))

print("\n[LightGBM Optuna - 50 trials]")
def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_log_train)
    pred = np.expm1(model.predict(X_test))
    return mean_absolute_percentage_error(y_orig_test, pred) * 100

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"LightGBM Best MAPE: {lgb_study.best_value:.2f}%")
print(f"Best params: {lgb_study.best_params}")

lgb_opt = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, n_jobs=-1, verbose=-1)
lgb_opt.fit(X_train, y_log_train)
y_pred_lgb_opt = np.expm1(lgb_opt.predict(X_test))
all_results.append(get_result('LightGBM (Optuna)', y_orig_test, y_pred_lgb_opt, y_orig_train))

print("\n[RandomForest Optuna - 30 trials]")
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42, 'n_jobs': -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_log_train)
    pred = np.expm1(model.predict(X_test))
    return mean_absolute_percentage_error(y_orig_test, pred) * 100

rf_study = optuna.create_study(direction='minimize')
rf_study.optimize(rf_objective, n_trials=30, show_progress_bar=False)
print(f"RandomForest Best MAPE: {rf_study.best_value:.2f}%")
print(f"Best params: {rf_study.best_params}")

rf_opt = RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)
rf_opt.fit(X_train, y_log_train)
y_pred_rf_opt = np.expm1(rf_opt.predict(X_test))
all_results.append(get_result('RandomForest (Optuna)', y_orig_test, y_pred_rf_opt, y_orig_train))

# ══════════════════════════════════════════════════════════════════════════════
# 14. 스태킹 앙상블 (Optuna 최적 모델 3개 → Ridge 메타 모델)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 14: 스태킹 앙상블")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Base 모델: Optuna 최적 파라미터 사용
base_models = [
    ('RF',  RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)),
    ('XGB', xgb.XGBRegressor(**xgb_study.best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)),
    ('LGB', lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, n_jobs=-1, verbose=-1)),
]

X_train_arr     = X_train.values
X_test_arr      = X_test.values
y_log_train_arr = y_log_train.values

# OOF 예측 행렬 (train용) / 테스트 예측 행렬 (폴드 평균)
oof_preds  = np.zeros((len(X_train), len(base_models)))
test_preds = np.zeros((len(X_test),  len(base_models)))

for j, (name, model) in enumerate(base_models):
    test_fold_preds = np.zeros((len(X_test), kf.n_splits))
    print(f"\n  [{name}] 5-Fold OOF 예측 중...")
    for i, (tr_idx, val_idx) in enumerate(kf.split(X_train_arr)):
        X_tr,  X_val = X_train_arr[tr_idx], X_train_arr[val_idx]
        y_tr         = y_log_train_arr[tr_idx]

        clone_model = type(model)(**model.get_params())
        clone_model.fit(X_tr, y_tr)

        oof_preds[val_idx, j]    = clone_model.predict(X_val)
        test_fold_preds[:, i]    = clone_model.predict(X_test_arr)

    test_preds[:, j] = test_fold_preds.mean(axis=1)
    oof_mape = mean_absolute_percentage_error(y_orig_train, np.expm1(oof_preds[:, j])) * 100
    print(f"    OOF MAPE: {oof_mape:.2f}%")

# 메타 모델 학습 (Ridge)
print("\n  [Ridge 메타 모델 학습]")
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y_log_train_arr)

# 테스트 최종 예측
y_pred_stack = np.expm1(meta_model.predict(test_preds))
all_results.append(get_result('Stacking (Ridge)', y_orig_test, y_pred_stack, y_orig_train))

print("\n메타 모델 계수 (각 base 모델 기여도):")
for (name, _), coef in zip(base_models, meta_model.coef_):
    print(f"  {name}: {coef:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 14-1. 스태킹 모델 피처 중요도 (Permutation Importance) - 수정본
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 14-1: 스태킹 모델 Permutation Importance 계산")
print("=" * 60)

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

# Scikit-learn 표준 인터페이스를 따르도록 클래스 수정
class StackingRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, base_models_with_names, meta_model):
        self.base_models_with_names = base_models_with_names
        self.meta_model = meta_model
        # Scikit-learn 내부 검증을 통과하기 위해 모델 타입 명시
        self._estimator_type = "regressor"

    def fit(self, X, y=None):
        """
        이미 베이스 모델과 메타 모델이 학습된 상태이므로, 
        형식상 fit 메서드만 정의하고 아무 작업도 하지 않습니다.
        """
        return self

    def predict(self, X):
        # 입력 데이터가 DataFrame인 경우 values로 변환 (속도 및 호환성)
        if hasattr(X, "values"):
            X_input = X.values
        else:
            X_input = X
            
        meta_features = np.zeros((len(X_input), len(self.base_models_with_names)))
        for i, (name, model) in enumerate(self.base_models_with_names):
            meta_features[:, i] = model.predict(X_input)
        
        return self.meta_model.predict(meta_features)

# 1. 학습이 완료된 최적 모델들로 래퍼 생성
fitted_base_models = [
    ('RF', rf_opt),
    ('XGB', xgb_opt),
    ('LGB', lgb_opt)
]
stacking_wrapper = StackingRegressorWrapper(fitted_base_models, meta_model)

# 2. 순열 중요도 계산
print("순열 중요도 계산 중... (데이터 양에 따라 시간이 소요될 수 있습니다)")
# 주의: 만약 여기서도 에러가 난다면 n_jobs=1로 변경해 보세요.
perm_importance = permutation_importance(
    stacking_wrapper, X_test, y_log_test, 
    n_repeats=10, random_state=42, n_jobs=-1
)

# 3. 결과 정리 및 시각화
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values(by='Importance', ascending=False)

print("\n스태킹 모델 Permutation Importance (Top 10):")
print(perm_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 8))
top_20_perm = perm_df.head(20).sort_values(by='Importance', ascending=True)
ax.barh(top_20_perm['Feature'], top_20_perm['Importance'], 
        xerr=top_20_perm['Std'], color='mediumpurple', alpha=0.8)
ax.set_title('Stacking Ensemble - Permutation Importance (Top 20)', fontsize=15)
ax.set_xlabel('Decrease in R² Score when Feature is Permuted')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '11_stacking_permutation_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 11 saved: 스태킹 모델 피처 중요도")

# ══════════════════════════════════════════════════════════════════════════════
# 15. 결과 비교 시각화
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 15: 결과 비교 시각화")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print("\n전체 모델 성능 비교:")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = [('R2', 'R² (높을수록 좋음)', 'steelblue'),
           ('RMSE', 'RMSE - 만원 (낮을수록 좋음)', 'salmon'),
           ('MAPE', 'MAPE % (낮을수록 좋음)', 'orange'),
           ('MAE', 'MAE - 만원 (낮을수록 좋음)', 'mediumseagreen')]

for ax, (metric, title, color) in zip(axes.flatten(), metrics):
    vals = results_df[metric]
    bars = ax.bar(results_df['Model'], vals, color=color, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontsize=13)
    ax.set_xticklabels(results_df['Model'], rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.suptitle('모델 성능 비교 (베이스라인 vs Optuna 최적화)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '06_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 06 saved: 모델 성능 비교")

best_idx  = results_df['MAPE'].idxmin()
best_name = results_df.loc[best_idx, 'Model']
pred_map = {
    'RandomForest (Base)': y_pred_rf,
    'XGBoost (Base)': y_pred_xgb,
    'LightGBM (Base)': y_pred_lgb,
    'XGBoost (Optuna)': y_pred_xgb_opt,
    'LightGBM (Optuna)': y_pred_lgb_opt,
    'RandomForest (Optuna)': y_pred_rf_opt,
    'Stacking (Ridge)': y_pred_stack,
}
best_pred = pred_map[best_name]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_orig_test, best_pred, alpha=0.4, color='steelblue', s=20)
mn = min(y_orig_test.min(), best_pred.min())
mx = max(y_orig_test.max(), best_pred.max())
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
axes[0].set_xlabel('실제 환산가 (만원)')
axes[0].set_ylabel('예측 환산가 (만원)')
axes[0].set_title(f'예측 vs 실제 ({best_name})', fontsize=13)
residuals = y_orig_test.values - best_pred
axes[1].scatter(best_pred, residuals, alpha=0.4, color='salmon', s=20)
axes[1].axhline(0, color='black', linewidth=1.5)
axes[1].set_xlabel('예측값 (만원)')
axes[1].set_ylabel('잔차 (실제 - 예측)')
axes[1].set_title(f'잔차 플롯 ({best_name})', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '07_best_model_pred_residual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 07 saved: 최적 모델 예측 vs 실제 / 잔차")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (study, name) in zip(axes, [(xgb_study,'XGBoost'), (lgb_study,'LightGBM'), (rf_study,'RandomForest')]):
    vals = [t.value for t in study.trials if t.value is not None]
    running_best = np.minimum.accumulate(vals)
    ax.plot(vals, alpha=0.5, color='lightblue', label='Trial MAPE')
    ax.plot(running_best, color='steelblue', linewidth=2, label='Best MAPE')
    ax.set_title(f'{name} Optuna 최적화 이력', fontsize=12)
    ax.set_xlabel('Trial')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '08_optuna_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 08 saved: Optuna 최적화 이력")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
models_opt = [
    ('XGBoost (Optuna)', xgb_opt, 'feature_importances_'),
    ('LightGBM (Optuna)', lgb_opt, 'feature_importances_'),
    ('RandomForest (Optuna)', rf_opt, 'feature_importances_'),
]
for ax, (name, model, attr) in zip(axes, models_opt):
    imp = pd.Series(getattr(model, attr), index=X.columns).sort_values(ascending=True).tail(15)
    imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'{name} 피처 중요도 (Top 15)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '09_feature_importances_optuna.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 09 saved: Optuna 모델 피처 중요도")

# ══════════════════════════════════════════════════════════════════════════════
# 15. 피처 박스플롯 - 주요 범주형 변수
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col in zip(axes, ['House_Type', 'Rent_Type', 'Orientation']):
    if col in df.columns:
        df.boxplot(column='Price_Hwansan', by=col, ax=ax, vert=True)
        ax.set_title(f'{col}별 환산가 분포')
        ax.set_xlabel(col)
        ax.set_ylabel('환산가 (만원)')
plt.suptitle('범주형 변수별 환산가 분포')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '10_categorical_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10 saved: 범주형 변수 박스플롯")

# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════════════════════════════
results_df.to_csv(os.path.join(BASE_DIR, 'model_results.csv'), index=False)
vif_df.to_csv(os.path.join(BASE_DIR, 'vif_results.csv'), index=False)

print("\n" + "=" * 60)
print("모든 분석 완료!")
print("=" * 60)
print(f"\n최고 성능 모델: {best_name}")
print(f"  R²  : {results_df.loc[best_idx,'R2']:.4f}")
print(f"  RMSE: {results_df.loc[best_idx,'RMSE']:.2f} 만원")
print(f"  MAPE: {results_df.loc[best_idx,'MAPE']:.2f}%")
print(f"\n저장된 그래프: {SAVE_DIR}")
