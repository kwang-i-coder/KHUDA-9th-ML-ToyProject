import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from rename import rename_map
# 1. 데이터 로드
try:
    df = pd.read_csv('with_orientation_final.csv', encoding='utf-8')
except Exception:
    df = pd.read_csv('with_orientation_final.csv', encoding='cp949', errors='replace')


df = df.rename(columns=rename_map)

# Cluster 값이 -1인 행 제거
df = df[df['Cluster_Label'] != -1]

# 2. 범주형 값 영문 변환
rent_map = {'월세': 'Monthly', '전세': 'Jeonse'}
house_map = {
    '다가구': 'Multi_Family', '다세대': 'Multi_Unit',
    '단독': 'Single_House', '연립': 'Row_House', '연립다세대': 'Villa'
}
ori_map = {
    '남향': 'South', '동향': 'East', '서향': 'West', '북향': 'North',
    '남동향': 'South_East', '남서향': 'South_West', 'Unknown': 'Unknown'
}

df['Rent_Type']   = df['Rent_Type'].map(rent_map).fillna('Other_Rent_Type')
df['House_Type']  = df['House_Type'].map(house_map).fillna('Other_House_Type')
df['Orientation'] = df['Orientation'].map(ori_map).fillna('Other_Ori')

# 3. 데이터 전처리
df['Deposit']      = pd.to_numeric(df['Deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Monthly_Rent'] = pd.to_numeric(df['Monthly_Rent'], errors='coerce').fillna(0)
df['Price_Hwansan'] = df['Deposit'] + (df['Monthly_Rent'] * 100)

df['Build_Year'] = pd.to_numeric(df['Build_Year'], errors='coerce')
df['House_Age']  = 2026 - df['Build_Year'].fillna(df['Build_Year'].median())

# 비정상 데이터 필터링
df = df[df['Price_Hwansan'] > 100]

# 4. 분석용 데이터 정제
drop_features = (
    [c for c in df.columns if '_Dist' in c] +
    ['Road_Name', 'Lat', 'Lon', 'Cluster_Label', 'Cluster', 'Cluster_Clean', 'Cluster_Name',
     'Build_Year', 'School_Walk_Dist', 'Price_Hwansan_Raw']
)
df_clean = df.drop(columns=drop_features, errors='ignore')

# fillna 전략 분리
count_score_cols = [c for c in df_clean.columns if any(k in c for k in ['Count', 'Score', 'Sum'])]
numeric_cols     = df_clean.select_dtypes(include=[np.number]).columns.tolist()
other_num_cols   = [c for c in numeric_cols if c not in count_score_cols and c != 'Price_Hwansan']

df_clean[count_score_cols] = df_clean[count_score_cols].fillna(0)
df_clean[other_num_cols]   = df_clean[other_num_cols].fillna(df_clean[other_num_cols].median())

# 5. 원-핫 인코딩 (기준 컬럼 명시적 제거)
df_final = pd.get_dummies(df_clean, columns=['House_Type', 'Rent_Type', 'Orientation'], drop_first=False)
baseline_cols = ['House_Type_Other_House_Type', 'Rent_Type_Other_Rent_Type', 'Orientation_Unknown']
df_final = df_final.drop(columns=[c for c in baseline_cols if c in df_final.columns])

# 6. VIF 계산
X = df_final.drop(columns=['Price_Hwansan', 'Deposit', 'Monthly_Rent'], errors='ignore').astype(float)

vif_df = pd.DataFrame()
vif_df['Feature'] = X.columns
vif_df['VIF']     = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# 무한대 값 처리
vif_df['VIF_Plot'] = vif_df['VIF'].replace([np.inf, -np.inf], 500)

print(vif_df.sort_values(by='VIF_Plot', ascending=False).to_string(index=False))

# 7. 시각화
plt.figure(figsize=(12, 10))
sns.barplot(x='VIF_Plot', y='Feature',
            data=vif_df.sort_values(by='VIF_Plot', ascending=False),
            palette='magma')
plt.axvline(x=10, color='red', linestyle='--', label='Threshold (10)')
plt.title('VIF Analysis')
plt.xlabel('VIF Value')
plt.legend()
plt.tight_layout()
plt.savefig('vif_analysis.png', dpi=150)
plt.show()

# 최종 데이터 저장
df_final.to_csv('fully_english_processed_data.csv', index=False)
print("\n최종 피처 리스트:")
print(X.columns.tolist())