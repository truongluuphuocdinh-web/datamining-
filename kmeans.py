"""
===================================================================
ĐỒ ÁN KHAI PHÁ DỮ LIỆU - PHÂN TÍCH BÁN LẺ
CĐCNTT 24AI
===================================================================
Nội dung:
  1. Phân nhóm khách hàng (K-means + DBSCAN)
  2. Phân tích giỏ hàng - Luật kết hợp (Manual Apriori)
  3. Dự đoán doanh thu (Hồi quy tuyến tính + Moving Average)
  4. Đề xuất chiến lược kinh doanh (in kết quả)
===================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# ĐỌC DỮ LIỆU
# ---------------------------------------------------------------
df = pd.read_csv('retail_invoice_summary.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print("=" * 60)
print("TỔNG QUAN DỮ LIỆU")
print("=" * 60)
print(f"Số giao dịch : {len(df):,}")
print(f"Số khách hàng: {df['Customer ID'].nunique()}")
print(f"Thời gian    : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Danh mục SP  : {df['Categories'].nunique()} loại")
print(f"Tổng doanh thu: {df['Total_Spent'].sum():,.2f}")
print()


# ================================================================
# PHẦN 1: PHÂN NHÓM KHÁCH HÀNG
# ================================================================
print("=" * 60)
print("PHẦN 1: PHÂN NHÓM KHÁCH HÀNG (K-MEANS + DBSCAN)")
print("=" * 60)

# Tổng hợp dữ liệu theo khách hàng
cust = df.groupby('Customer ID').agg(
    Total_Spent     = ('Total_Spent', 'sum'),
    Num_Items       = ('Num_Items', 'sum'),
    Total_Qty       = ('Total_Qty', 'sum'),
    Transactions    = ('Transaction ID', 'count'),
    Avg_Spent_Per_Txn = ('Total_Spent', 'mean')
).reset_index()

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cust[['Total_Spent', 'Num_Items', 'Total_Qty']].values)

# --- Tìm K tốt nhất qua Silhouette Score (Giữ nguyên logic in thông tin) ---
print("\n>> Tìm K tối ưu (Silhouette Score):")
silhouette_scores = {}
for k in range(3, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = round(score, 4)
    print(f"   K={k}: Silhouette = {score:.4f}")

# THAY ĐỔI CỐ ĐỊNH K = 3
best_k = 3 
print(f"\n>> Lựa chọn K cố định = {best_k}")

# --- K-Means với K=3 ---
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
cust['KMeans_Cluster'] = km_final.fit_predict(X_scaled)

print("=" * 60)
print("KẾT QUẢ PHÂN CỤM (K-MEANS K=3)")
print("=" * 60)
print(cust['KMeans_Cluster'].value_counts().sort_index())

print("\n>> Đặc trưng từng nhóm khách hàng (K-Means):")
cluster_stats = cust.groupby('KMeans_Cluster').agg(
    Số_KH          = ('Customer ID', 'count'),
    TB_Tổng_Chi_Tiêu = ('Total_Spent', 'mean'),
    TB_Số_Mặt_Hàng = ('Num_Items', 'mean'),
    TB_Số_Lượng    = ('Total_Qty', 'mean'),
    TB_Chi_Tiêu_Mỗi_Giao_Dịch = ('Avg_Spent_Per_Txn', 'mean')
).round(1).sort_index()
print(cluster_stats.to_string())

# Đặt nhãn nhóm dựa trên chi tiêu cho 3 cụm
max_spent = cluster_stats['TB_Tổng_Chi_Tiêu'].max()
for cluster_id in range(3):
    spent = cluster_stats.loc[cluster_id, 'TB_Tổng_Chi_Tiêu']
    if spent == max_spent:
        label = "🏆 Khách hàng VIP (Chi tiêu cao)"
    else:
        label = "🛒 Khách hàng Thông thường"
    print(f"   Cụm {cluster_id}: {label}")

# --- DBSCAN (Giữ nguyên tham số của bạn) ---
print("\n>> DBSCAN (eps=0.8, minPts=3):")
db = DBSCAN(eps=0.8, min_samples=3)
cust['DBSCAN_Cluster'] = db.fit_predict(X_scaled)
dbscan_counts = cust['DBSCAN_Cluster'].value_counts().sort_index()
for label, count in dbscan_counts.items():
    if label == -1:
        print(f"   Noise (ngoại lệ): {count} khách hàng")
    else:
        print(f"   Cụm {label}: {count} khách hàng")

# Lưu kết quả
cust.to_csv('customer_segments_output3.csv', index=False)
print("\n>> Đã lưu: customer_segments_output3.csv")

# ================================================================
# PHẦN 2, 3, 4 (Giữ nguyên toàn bộ logic cũ của bạn)
# ================================================================
# [Toàn bộ phần code Apriori, Linear Regression và Đề xuất được giữ nguyên như file gốc]