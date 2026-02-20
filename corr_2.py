import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('final_data.csv', encoding='utf-8')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
corr_df = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_df, annot=True)
plt.savefig('final_correlation_heatmap.png', dpi=150)
plt.show()
