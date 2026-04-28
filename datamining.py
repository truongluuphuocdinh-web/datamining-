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

# --- Tìm K tốt nhất qua Silhouette Score ---
print("\n>> Tìm K tối ưu (Silhouette Score):")
silhouette_scores = {}
for k in range(3, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = round(score, 4)
    print(f"   K={k}: Silhouette = {score:.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n>> K tối ưu = {best_k} (Silhouette Score = {silhouette_scores[best_k]})")
print("   (Silhouette gần 1.0 = phân cụm tốt; gần 0 = chồng lấp; âm = sai cụm)")

# --- K-Means với K tối ưu ---
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
).round(1)
print(cluster_stats.to_string())

# Đặt nhãn nhóm dựa trên chi tiêu
for cluster_id in range(best_k):
    spent = cluster_stats.loc[cluster_id, 'TB_Tổng_Chi_Tiêu']
    if spent == cluster_stats['TB_Tổng_Chi_Tiêu'].max():
        label = "🏆 Khách hàng VIP (Chi tiêu cao)"
    else:
        label = "🛒 Khách hàng Thông thường"
    print(f"   Cụm {cluster_id}: {label}")

# --- DBSCAN ---
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
cust.to_csv('customer_segments_output.csv', index=False)
print("\n>> Đã lưu: customer_segments_output.csv")


# ================================================================
# PHẦN 2: PHÂN TÍCH GIỎ HÀNG (APRIORI - LUẬT KẾT HỢP)
# ================================================================
print("\n" + "=" * 60)
print("PHẦN 2: PHÂN TÍCH GIỎ HÀNG (ASSOCIATION RULES)")
print("=" * 60)

# Tạo giỏ hàng theo khách hàng-tháng
df['YearMonth'] = df['Date'].dt.to_period('M')
baskets = df.groupby(['Customer ID', 'YearMonth'])['Categories'].apply(list)
total_baskets = len(baskets)

print(f"\n>> Tổng số giỏ hàng (KH-tháng): {total_baskets}")
print(f"   Trung bình items/giỏ: {baskets.apply(len).mean():.2f}")

# Đếm tần suất
pair_counts   = defaultdict(int)
single_counts = defaultdict(int)

for basket in baskets:
    cats = list(set(basket))
    for c in cats:
        single_counts[c] += 1
    for a, b in combinations(sorted(cats), 2):
        pair_counts[(a, b)] += 1

# Support đơn
print("\n>> Support từng danh mục (L1):")
sup1 = {k: round(v / total_baskets, 3) for k, v in single_counts.items()}
for cat, sup in sorted(sup1.items(), key=lambda x: -x[1]):
    print(f"   {cat:<40} support = {sup:.3f}  ({int(sup*total_baskets)} lần)")

# Sinh luật kết hợp
MIN_SUPPORT    = 0.50
MIN_CONFIDENCE = 0.70

rules = []
for (a, b), cnt in pair_counts.items():
    sup_ab  = cnt / total_baskets
    sup_a   = single_counts[a] / total_baskets
    sup_b   = single_counts[b] / total_baskets

    if sup_ab < MIN_SUPPORT:
        continue

    # A → B
    conf_ab = sup_ab / sup_a
    lift_ab = conf_ab / sup_b
    if conf_ab >= MIN_CONFIDENCE:
        rules.append({'Antecedent (A)': a, 'Consequent (B)': b,
                      'Support': round(sup_ab, 3),
                      'Confidence': round(conf_ab, 3),
                      'Lift': round(lift_ab, 3)})

    # B → A
    conf_ba = sup_ab / sup_b
    lift_ba = conf_ba / sup_a
    if conf_ba >= MIN_CONFIDENCE:
        rules.append({'Antecedent (A)': b, 'Consequent (B)': a,
                      'Support': round(sup_ab, 3),
                      'Confidence': round(conf_ba, 3),
                      'Lift': round(lift_ba, 3)})

rules_df = (pd.DataFrame(rules)
              .drop_duplicates()
              .sort_values('Lift', ascending=False)
              .reset_index(drop=True))

print(f"\n>> Luật kết hợp tìm được: {len(rules_df)} luật")
print(f"   (min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE})")
print("\n>> Top 10 luật mạnh nhất (theo Lift):")
print(rules_df.head(10).to_string(index=False))
print("\n>> Giải thích chỉ số Lift:")
print("   Lift > 1 → A và B thực sự liên quan (mua A tăng xác suất mua B)")
print("   Lift = 1 → Ngẫu nhiên, không liên quan")
print("   Lift < 1 → A và B ít khi đi cùng nhau")

rules_df.to_csv('association_rules_output3.csv', index=False)
print("\n>> Đã lưu: association_rules_output.csv")


# ================================================================
# PHẦN 3: DỰ ĐOÁN DOANH THU
# ================================================================
print("\n" + "=" * 60)
print("PHẦN 3: DỰ ĐOÁN DOANH THU")
print("=" * 60)

monthly = (df.groupby(df['Date'].dt.to_period('M'))['Total_Spent']
             .sum()
             .reset_index())
monthly.columns = ['Month', 'Revenue']
monthly['MA3']   = monthly['Revenue'].rolling(3).mean().round(2)
monthly['MA6']   = monthly['Revenue'].rolling(6).mean().round(2)
monthly['month_num'] = range(len(monthly))

# Hồi quy tuyến tính
lr = LinearRegression()
lr.fit(monthly[['month_num']].values, monthly['Revenue'].values)
monthly['LR_Pred'] = lr.predict(monthly[['month_num']].values).round(2)

next_month_pred_lr  = lr.predict([[len(monthly)]])[0]
next_month_pred_ma3 = monthly['MA3'].iloc[-1]

monthly['Month_str'] = monthly['Month'].astype(str)
monthly.to_csv('monthly_revenue_output3.csv', index=False)

print("\n>> Doanh thu theo tháng (12 tháng gần nhất):")
print(monthly[['Month_str', 'Revenue', 'MA3', 'MA6', 'LR_Pred']].tail(12).to_string(index=False))
print(f"\n>> Dự đoán tháng tiếp theo:")
print(f"   Hồi quy tuyến tính (Linear Regression): {next_month_pred_lr:,.2f}")
print(f"   Trung bình trượt 3 tháng (MA3)        : {next_month_pred_ma3:,.2f}")
print(f"   R² của mô hình LR: {lr.score(monthly[['month_num']].values, monthly['Revenue'].values):.4f}")

print("\n>> Đã lưu: monthly_revenue_output.csv")


# ================================================================
# PHẦN 4: ĐỀ XUẤT CHIẾN LƯỢC KINH DOANH
# ================================================================
print("\n" + "=" * 60)
print("PHẦN 4: ĐỀ XUẤT CHIẾN LƯỢC KINH DOANH")
print("=" * 60)

print("""
A. SẮP XẾP KỆ HÀNG DỰA TRÊN LUẬT KẾT HỢP:
   • Đặt kệ "Đồ gia dụng điện" (EHE) gần kệ "Thực phẩm" và "Đồ uống"
     → Lift ~1.01-1.02 cho thấy mối liên quan tích cực
   • Nhóm "Bánh ngọt" (Patisserie) gần "Đồ uống" (Beverages)
   • Khu vực thanh toán: trưng bày combo sản phẩm từ các cặp luật mạnh

B. CHƯƠNG TRÌNH COMBO / KHUYẾN MÃI:
   Nhóm VIP (Cụm 0 - Chi tiêu cao):
   • Thẻ thành viên VIP với ưu đãi tích điểm 2x
   • Bundle cao cấp: Thiết bị điện + Nội thất + tặng bảo hành mở rộng
   • Ưu tiên giao hàng nhanh & dịch vụ cá nhân hóa

   Nhóm Thông thường (Cụm 1):
   • Combo tiết kiệm: Thực phẩm + Sữa + Bánh ngọt
   • Chương trình "Mua 3 tặng 1" cho các danh mục phổ biến
   • Voucher giảm giá khi đạt ngưỡng chi tiêu tháng

C. DỰA TRÊN DỰ ĐOÁN DOANH THU:
   • Doanh thu dự báo giảm nhẹ so với đỉnh tháng 12/2024
   • Đề xuất chạy chiến dịch khuyến mãi đầu năm (tháng 1-2)
   • Tập trung upsell cho nhóm VIP để duy trì doanh thu
""")

print("=" * 60)
print("PHÂN TÍCH HOÀN TẤT - Kiểm tra các file CSV output!")
print("=" * 60)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# VẼ BIỂU ĐỒ DỰ BÁO DOANH THU (REGRESSION)
# ---------------------------------------------------------------
plt.figure(figsize=(12, 6))

# 1. Vẽ đường doanh thu thực tế
plt.plot(monthly['Month_str'], monthly['Revenue'], marker='o', label='Doanh thu thực tế', color='#1f77b4', linewidth=2)

# 2. Vẽ đường dự báo của Hồi quy tuyến tính (Linear Regression)
plt.plot(monthly['Month_str'], monthly['LR_Pred'], linestyle='--', label='Đường xu hướng (Regression)', color='#d62728')

# 3. Vẽ đường trung bình trượt (MA3) để thấy biến động ngắn hạn
plt.plot(monthly['Month_str'], monthly['MA3'], label='Trung bình trượt 3 tháng (MA3)', color='#2ca02c', alpha=0.7)

# Cấu hình biểu đồ
plt.title('PHÂN TÍCH VÀ DỰ BÁO DOANH THU THEO THÁNG', fontsize=15, fontweight='bold')
plt.xlabel('Tháng', fontsize=12)
plt.ylabel('Doanh thu (VND)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()

# Lưu và hiển thị
plt.tight_layout()
plt.savefig('revenue_forecast_chart.png', dpi=300)
plt.show()

print(">> Đã xuất biểu đồ dự báo doanh thu: revenue_forecast_chart.png")