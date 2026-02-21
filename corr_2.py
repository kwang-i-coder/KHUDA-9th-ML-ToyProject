import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

corr_df = df.corr()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(20, 20))
sns.heatmap(corr_df, annot=True)
plt.savefig('cctv_lamp_correlation.png', dpi=150)
plt.show()
