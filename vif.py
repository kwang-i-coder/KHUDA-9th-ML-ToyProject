import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from rename import rename_map

# 1. 데이터 로드 및 기본 정제
try:
    df = pd.read_csv('with_orientation_final.csv', encoding='utf-8')
except Exception:
    df = pd.read_csv('with_orientation_final.csv', encoding='cp949', errors='replace')

df = df.rename(columns=rename_map)
df = df[df['Cluster_Label'] != -1] # 이상치 제거

# 2. 하버사인 거리 계산 (이전 코드의 장점 유지)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

gate_lat, gate_lon = 37.2470, 127.0736 
if 'Lat' in df.columns and 'Lon' in df.columns:
    df['Dist_to_Univ_km'] = haversine(df['Lat'], df['Lon'], gate_lat, gate_lon)

# 3. 범주형 변환 및 파생변수 생성
rent_map = {'월세': 'Monthly', '전세': 'Jeonse'}
house_map = {'다가구': 'Multi_Family', '다세대': 'Multi_Unit', '단독': 'Single_House', '연립': 'Row_House', '연립다세대': 'Villa'}
ori_map = {'남향': 'South', '동향': 'East', '서향': 'West', '북향': 'North', '남동향': 'South_East', '남서향': 'South_West', 'Unknown': 'Unknown'}

df['Rent_Type'] = df['Rent_Type'].map(rent_map).fillna('Other_Rent')
df['House_Type'] = df['House_Type'].map(house_map).fillna('Other_House')
df['Orientation'] = df['Orientation'].map(ori_map).fillna('Other_Ori')

df['Deposit'] = pd.to_numeric(df['Deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Monthly_Rent'] = pd.to_numeric(df['Monthly_Rent'], errors='coerce').fillna(0)
df['Price_Hwansan'] = df['Deposit'] + (df['Monthly_Rent'] * 100)
df['House_Age'] = 2026 - pd.to_numeric(df['Build_Year'], errors='coerce').fillna(df['Build_Year'].median())

# [중요] 인프라 합계 점수 생성 후 개별 항목은 나중에 제거하여 다중공선성 방지
infra_cols = ['Conv_Count', 'Bank_Count', 'Hosp_Count', 'Cafe_Count', 'Rest_Count']
df['Life_Infra_Score'] = df[infra_cols].fillna(0).sum(axis=1)

# 4. 분석용 데이터 정제 (불필요 피처 제거)
# VIF 계산 시 '합계' 변수와 '개별' 변수가 동시에 들어가면 안 됨
drop_features = (
    [c for c in df.columns if '_Dist' in c] + infra_cols +
    ['Road_Name', 'Lat', 'Lon', 'Cluster_Label', 'Cluster', 'Cluster_Clean', 'Cluster_Name',
     'Build_Year', 'School_Walk_Dist', 'Price_Hwansan_Raw', 'Pharm_Count', 'Parking_Count',
     'Deposit', 'Monthly_Rent', 'Transit_Min'] # Target Leakage 및 중복 변수 제거
)
df_clean = df.drop(columns=drop_features, errors='ignore')

# 결측치 처리
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# 5. 원-핫 인코딩
df_final = pd.get_dummies(df_clean, columns=['House_Type', 'Rent_Type', 'Orientation'], drop_first=True)

# 6. VIF 계산
# Price_Hwansan(Target) 제외
X = df_final.drop(columns=['Price_Hwansan'], errors='ignore').astype(float)
X_vif = add_constant(X) # 상수항 추가 (VIF 계산의 정석)

vif_df = pd.DataFrame()
vif_df['Feature'] = X_vif.columns
vif_df['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

# 상수항 제외하고 결과 보기
vif_df = vif_df[vif_df['Feature'] != 'const']
vif_df['VIF_Plot'] = vif_df['VIF'].replace([np.inf, -np.inf], 100) # 시각화를 위해 상한선 조정

# 7. 시각화 및 저장
plt.figure(figsize=(10, 8))
sns.barplot(x='VIF_Plot', y='Feature', data=vif_df.sort_values(by='VIF', ascending=False), palette='viridis')
plt.axvline(x=10, color='red', linestyle='--', label='High Collinearity Threshold (10)')
plt.title('VIF Analysis for Rental Price Prediction')
plt.tight_layout()
plt.show()

print(vif_df.sort_values(by='VIF', ascending=False))