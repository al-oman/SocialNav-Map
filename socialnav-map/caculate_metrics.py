import pandas as pd
import numpy as np

from io import StringIO
    
# 读取CSV数据

df = pd.read_csv('/home/ubuntu/socialnav/InstructNav_r2r_lf/chaoxiang_mp3d.csv')

print("=== 使用Pandas计算结果 ===")
print(f"数据行数: {len(df)}")
print("\n各指标平均值:")

# 计算各指标平均值
metrics = {}
for column in df.columns:
    avg_value = df[column].mean()
    print(df[column].sum())
    metrics[column] = avg_value
    print(f"{column}: {avg_value:.6f}")

result = 0.4*metrics['success'] + 0.3*metrics['spl'] + 0.3*metrics['psc']
result1 = 0.4*metrics['success'] + 0.2*metrics['spl'] + 0.2*metrics['psc'] +0.2*(1-metrics['human_collision'])
print(result)
print(result1)