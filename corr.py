import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rename import rename_map

# 1. 데이터 로드
try:
    df = pd.read_csv('with_orientation_final.csv', encoding='utf-8')
except Exception:
    df = pd.read_csv('with_orientation_final.csv', encoding='cp949', errors='replace')


df = df.rename(columns=rename_map)

# Cluster 값이 -1인 행 제거
df = df[df['Cluster_Label'] != -1]

# 2. 파생 변수 생성
df['Deposit']       = pd.to_numeric(df['Deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Monthly_Rent']  = pd.to_numeric(df['Monthly_Rent'], errors='coerce').fillna(0)
df['Price_Hwansan'] = df['Deposit'] + (df['Monthly_Rent'] * 100)

df['Build_Year'] = pd.to_numeric(df['Build_Year'], errors='coerce')
df['House_Age']  = 2026 - df['Build_Year'].fillna(df['Build_Year'].median())

# 비정상 데이터 필터링 (보증금/월세 모두 0인 명백한 오류만 제거)
df = df[df['Price_Hwansan'] > 0]

# 3. 상관관계 분석용 수치형 변수 선택
numeric_features = [
    'Price_Hwansan', 'Area_m2', 'House_Age', 'School_Walk_Min', 'Transit_Min',
    'Conv_Count', 'Bank_Count', 'Hosp_Count', 'Pharm_Count',
    'Cafe_Count', 'Rest_Count', 'Mart_Count', 'Parking_Count'
]
df_corr = df[numeric_features].fillna(0)

# 4. 상관계수 행렬 계산
corr_matrix = df_corr.corr()

# 5. 히트맵 시각화
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='RdBu_r',
    fmt='.2f',
    center=0,
    linewidths=0.5
)
plt.title('Correlation Heatmap: Feature Associations', fontsize=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()