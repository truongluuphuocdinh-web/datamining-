"""
==============================================================
DATA REDUCTION - Retail Store Sales
Phục vụ: Clustering, Association Rules, Revenue Forecasting
==============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys

# ── Đọc dữ liệu đã làm sạch ──────────────────────────────────
df = pd.read_csv('retail_cleaned.csv', sep=';')
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], dayfirst=True)

print("=" * 60)
print("PHẦN 3: GIẢM DỮ LIỆU (DATA REDUCTION)")
print("=" * 60)

# Ghi nhớ trạng thái BAN ĐẦU để so sánh
mem_before  = df.memory_usage(deep=True).sum() / 1024      # KB
rows_before = len(df)
cols_before = len(df.columns)
print(f"\n[TRƯỚC] {rows_before} dòng × {cols_before} cột | Bộ nhớ: {mem_before:.1f} KB")


# ==============================================================
# BƯỚC 1: FEATURE SELECTION — Giữ lại cột cần thiết
# ==============================================================
print("\n" + "-" * 60)
print("BƯỚC 1: FEATURE SELECTION")
print("-" * 60)

# Cột giữ lại phục vụ 3 bài toán
KEEP_COLS = [
    'Transaction ID',    # Mã hóa đơn  → Association Rules
    'Customer ID',       # Mã khách    → Clustering
    'Category',          # Danh mục    → Association Rules / Clustering
    'Item',              # Sản phẩm    → Association Rules
    'Total Spent',       # Tổng tiền   → Revenue Forecasting / Clustering
    'Transaction Date',  # Ngày mua    → Revenue Forecasting
    'Quantity',          # Số lượng    → Clustering / Forecasting
    'Discount Applied',  # Giảm giá    → Clustering
]

# Cột bị loại và lý do
dropped_cols = [c for c in df.columns if c not in KEEP_COLS]
drop_reasons = {
    'Price Per Unit': 'Có thể suy ra từ Total Spent / Quantity; không cần trực tiếp',
    'Payment Method': 'Không liên quan đến hành vi mua sắm trong 3 bài toán mục tiêu',
    'Location':       'Chỉ có 2 giá trị (Online/Offline), ít tách biệt phân cụm',
}

print("\nCác cột bị loại:")
for c in dropped_cols:
    print(f"  ✗ {c:20s} — {drop_reasons.get(c, 'Không cần thiết')}")

df_selected = df[KEEP_COLS].copy()
print(f"\n✓ Giữ lại {len(KEEP_COLS)} / {cols_before} cột")


# ==============================================================
# BƯỚC 2: NUMEROSITY REDUCTION — Gom theo hóa đơn
# ==============================================================
print("\n" + "-" * 60)
print("BƯỚC 2: NUMEROSITY REDUCTION  (GroupBy Transaction ID)")
print("-" * 60)

# Mỗi Transaction ID → 1 dòng tóm tắt
invoice_summary = (
    df_selected
    .groupby(['Transaction ID', 'Customer ID', 'Transaction Date'])
    .agg(
        Total_Spent    = ('Total Spent',   'sum'),
        Num_Items      = ('Item',          'count'),       # số dòng (mặt hàng)
        Total_Qty      = ('Quantity',      'sum'),
        Discount_Applied = ('Discount Applied', 'max'),   # True nếu bất kỳ mặt hàng có giảm giá
        Categories     = ('Category',      lambda x: '|'.join(sorted(x.unique()))),
        Items_List     = ('Item',          lambda x: '|'.join(sorted(x.unique()))),
    )
    .reset_index()
    .rename(columns={'Transaction Date': 'Date'})
)

print(f"\n✓ Sau GroupBy: {len(invoice_summary)} hóa đơn (từ {rows_before} dòng gốc)")
print(f"  Mỗi hóa đơn gồm: Transaction ID, Customer ID, Date,")
print(f"  Total_Spent, Num_Items, Total_Qty, Discount_Applied,")
print(f"  Categories (pipe-separated), Items_List (pipe-separated)")
print(f"\nVí dụ 3 hóa đơn đầu:")
print(invoice_summary.head(3).to_string(index=False))


# ==============================================================
# BƯỚC 3: DATA COMPRESSION — Mã hóa cột phân loại
# ==============================================================
print("\n" + "-" * 60)
print("BƯỚC 3: DATA COMPRESSION  (Encoding)")
print("-" * 60)

# ── 3a: Label Encoding cho Category (ordinal-safe, 8 nhãn) ───
le_cat = LabelEncoder()
df_selected['Category_LE'] = le_cat.fit_transform(df_selected['Category'])
cat_map = dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_)))
print("\n3a. Label Encoding — Category:")
for k, v in cat_map.items():
    print(f"    {v} → {k}")

# ── 3b: Label Encoding cho Item (200 giá trị → One-Hot quá rộng) ─
le_item = LabelEncoder()
df_selected['Item_LE'] = le_item.fit_transform(df_selected['Item'])
print(f"\n3b. Label Encoding — Item: {df_selected['Item'].nunique()} giá trị → cột số nguyên")

# ── 3c: One-Hot Encoding cho Category (dùng khi cần sparse matrix) ─
cat_ohe = pd.get_dummies(df_selected['Category'], prefix='Cat').astype(np.int8)
df_ohe  = pd.concat([df_selected.drop(columns=['Category', 'Item']), cat_ohe], axis=1)
print(f"\n3c. One-Hot Encoding — Category: tạo {cat_ohe.shape[1]} cột nhị phân (int8)")

# ── 3d: Tối ưu kiểu dữ liệu (Downcasting) ────────────────────
df_compressed = df_ohe.copy()
df_compressed['Quantity']       = df_compressed['Quantity'].astype(np.int8)
df_compressed['Total Spent']    = pd.to_numeric(df_compressed['Total Spent'],  downcast='float')
df_compressed['Item_LE']        = pd.to_numeric(df_compressed['Item_LE'],      downcast='integer')
df_compressed['Category_LE']    = pd.to_numeric(df_compressed['Category_LE'],  downcast='integer')
df_compressed['Discount Applied'] = df_compressed['Discount Applied'].astype(np.int8)
print("\n3d. Downcasting kiểu dữ liệu: float64→float32, int64→int8/int16")


# ==============================================================
# SO SÁNH BỘ NHỚ TRƯỚC / SAU
# ==============================================================
print("\n" + "=" * 60)
print("SO SÁNH BỘ NHỚ & THUỘC TÍNH")
print("=" * 60)

mem_after_sel  = df_selected.memory_usage(deep=True).sum()  / 1024
mem_after_comp = df_compressed.memory_usage(deep=True).sum() / 1024

print(f"\n{'Giai đoạn':<35} {'Dòng':>8} {'Cột':>6} {'Bộ nhớ (KB)':>14}")
print("-" * 65)
print(f"{'Gốc (raw cleaned)':<35} {rows_before:>8} {cols_before:>6} {mem_before:>14.1f}")
print(f"{'Sau Feature Selection':<35} {len(df_selected):>8} {len(df_selected.columns):>6} {mem_after_sel:>14.1f}")
print(f"{'Sau Compression (OHE + cast)':<35} {len(df_compressed):>8} {len(df_compressed.columns):>6} {mem_after_comp:>14.1f}")
print(f"\n→ Tiết kiệm bộ nhớ tổng: {(1 - mem_after_comp/mem_before)*100:.1f}%")

# ==============================================================
# LƯU CÁC TẬP KẾT QUẢ
# ==============================================================
df_selected.to_csv('retail_feature_selected.csv', index=False)
invoice_summary.to_csv('retail_invoice_summary.csv', index=False)
df_compressed.to_csv('retail_compressed.csv', index=False)

print("\n" + "=" * 60)
print("ĐÃ LƯU 3 FILE:")
print("  1. retail_feature_selected.csv  — Sau Feature Selection")
print("  2. retail_invoice_summary.csv   — Sau Numerosity Reduction")
print("  3. retail_compressed.csv        — Sau Compression đầy đủ")
print("=" * 60)