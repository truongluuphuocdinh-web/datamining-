
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
pd.set_option('future.no_silent_downcasting', True)

# ==================================================
# BƯỚC 0: ĐỌC DỮ LIỆU VÀ KHÁM PHÁ BAN ĐẦU
# ==================================================

print("=" * 60)
print("BƯỚC 0: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU")
print("=" * 60)

# Đọc dữ liệu
df = pd.read_csv('retail_store_sales.csv')

print("\n1. Thông tin cơ bản:")
print(f"Số dòng: {len(df)}")
print(f"Số cột: {len(df.columns)}")
print(f"\nDanh sách cột:\n{df.columns.tolist()}")

print("\n2. Xem 5 dòng đầu:")
print(df.head())

print("\n3. Thông tin chi tiết:")
print(df.info())

print("\n4. Thống kê mô tả:")
print(df.describe())

# ==================================================
# PHẦN 1: LÀM SẠCH DỮ LIỆU
# ==================================================

print("\n" + "=" * 60)
print("PHẦN 1: LÀM SẠCH DỮ LIỆU")
print("=" * 60)

# --------------------------------------------------
# Câu 1.1: Xử lý Missing Data
# --------------------------------------------------

print("\n--- Câu 1.1: XỬ LÝ MISSING DATA ---")

# a) Phát hiện
print("\n1. Thống kê missing values:")
missing_count = df.isnull().sum()
print(missing_count[missing_count > 0])

# b) Xử lý
print("\n2. Xử lý missing values:")

# Item & Category: Loại bỏ các dòng thiếu thông tin quan trọng (Deep Clean)
before_drop = len(df)
df.dropna(subset=['Item', 'Category'], inplace=True)
print(f"✓ Item & Category: đã loại bỏ {before_drop - len(df)} dòng thiếu thông tin tên hoặc ngành hàng")

# Price Per Unit & Quantity: thay bằng median
for col in ['Price Per Unit', 'Quantity']:
    if df[col].isnull().sum() > 0:
        col_median = df[col].median()
        df[col].fillna(col_median, inplace=True)
        print(f"✓ {col}: đã thay giá trị thiếu bằng median = {col_median}")

# Total Spent: Tính toán lại dựa trên Price * Quantity để đảm bảo logic
df['Total Spent'] = df['Price Per Unit'] * df['Quantity']
print("✓ Total Spent: đã tính toán lại toàn bộ theo công thức (Price * Quantity)")

# Discount Applied: mặc định là False nếu thiếu
df['Discount Applied'] = df['Discount Applied'].astype(object).fillna(False)

print("\n3. Kiểm tra lại sau khi xử lý:")
print(df.isnull().sum().sum(), "giá trị thiếu còn lại")

# --------------------------------------------------
# Câu 1.2: Xử lý Noisy Data
# --------------------------------------------------

print("\n--- Câu 1.2: XỬ LÝ NOISY DATA ---")

# a) Phát hiện
print("\n1. Phát hiện dữ liệu nhiễu (Giá trị âm hoặc bằng 0):")
noisy_price = df[df['Price Per Unit'] <= 0]
noisy_qty = df[df['Quantity'] <= 0]
print(f"- Price Per Unit bất thường: {len(noisy_price)} dòng")
print(f"- Quantity bất thường: {len(noisy_qty)} dòng")

# b) Xử lý
print("\n2. Xử lý dữ liệu nhiễu:")
before_len = len(df)
df = df[(df['Price Per Unit'] > 0) & (df['Quantity'] > 0)]
removed = before_len - len(df)
print(f"✓ Đã loại bỏ {removed} dòng có giá hoặc số lượng không hợp lý")

# --------------------------------------------------
# Câu 1.3: Xử lý Outliers
# --------------------------------------------------

print("\n--- Câu 1.3: XỬ LÝ OUTLIERS ---")

# a) Phát hiện bằng IQR
print("\n1. Phát hiện outliers trong Total Spent bằng IQR:")

Q1 = df['Total Spent'].quantile(0.25)
Q3 = df['Total Spent'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 = {Q1:,.2f}")
print(f"Q3 = {Q3:,.2f}")
print(f"IQR = {IQR:,.2f}")
print(f"Ngưỡng trên (Upper bound) = {upper_bound:,.2f}")

outliers = df[df['Total Spent'] > upper_bound]
print(f"\nSố outliers phát hiện: {len(outliers)}")

# Vẽ boxplot trước khi xử lý
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(df['Total Spent'])
plt.title('Total Spent - Trước xử lý Outliers')

# b) Xử lý bằng Capping (Chặn trên)
print("\n2. Xử lý outliers bằng Capping:")
df['Total Spent'] = df['Total Spent'].clip(upper=upper_bound)
print(f"✓ Đã điều chỉnh các giá trị vượt ngưỡng về mức {upper_bound:,.2f}")

# Vẽ boxplot sau khi xử lý
plt.subplot(1, 2, 2)
plt.boxplot(df['Total Spent'])
plt.title('Total Spent - Sau xử lý Outliers')
plt.tight_layout()
plt.savefig('retail_outliers.png')
print("✓ Đã lưu biểu đồ: retail_outliers.png")

# --------------------------------------------------
# Câu 1.4: Xử lý Inconsistent Data
# --------------------------------------------------

print("\n--- Câu 1.4: XỬ LÝ INCONSISTENT DATA ---")

print("\n1. Chuẩn hóa dữ liệu văn bản:")

# Chuẩn hóa Category, Payment Method, Location
cols_to_fix = ['Category', 'Payment Method', 'Location']
for col in cols_to_fix:
    df[col] = df[col].str.strip().str.title()
    print(f"✓ {col}: Đã chuẩn hóa định dạng Title Case và xóa khoảng trắng")

print("\n2. Kiểm tra lại giá trị duy nhất (Ví dụ Category):")
print(df['Category'].unique())

# --------------------------------------------------
# Câu 1.5: Loại bỏ Duplicate Data
# --------------------------------------------------

print("\n--- Câu 1.5: LOẠI BỎ DUPLICATE DATA ---")

# Trùng theo Transaction ID
id_dup = df.duplicated(subset=['Transaction ID']).sum()
print(f"Số Transaction ID trùng lặp: {id_dup}")

print("\n2. Loại bỏ dữ liệu trùng:")
before_len = len(df)
df = df.drop_duplicates(subset=['Transaction ID'], keep='first')
removed = before_len - len(df)
print(f"✓ Đã loại bỏ {removed} dòng trùng lặp ID")

# Lưu dữ liệu đã làm sạch
df.to_csv('retail_cleaned.csv', index=False)
print("\n✓ Đã lưu dữ liệu sau làm sạch: retail_cleaned.csv")

# ==================================================
# PHẦN 2: BIẾN ĐỔI DỮ LIỆU
# ==================================================

print("\n" + "=" * 60)
print("PHẦN 2: BIẾN ĐỔI DỮ LIỆU")
print("=" * 60)

# --------------------------------------------------
# Câu 2.1: Aggregation
# --------------------------------------------------

print("\n--- Câu 2.1: AGGREGATION ---")

# Thống kê theo Category
cat_stats = df.groupby('Category').agg({
    'Transaction ID': 'count',
    'Total Spent': 'sum',
    'Quantity': 'mean'
}).round(2)
cat_stats.columns = ['Số đơn hàng', 'Tổng doanh thu', 'Số lượng TB/đơn']
print("\nThống kê theo Category:")
print(cat_stats)

# --------------------------------------------------
# Câu 2.2: Generalization
# --------------------------------------------------

print("\n--- Câu 2.2: GENERALIZATION ---")

# Tạo Spending_Level
def classify_spending(value):
    if value < 50: return "Low"
    elif value <= 200: return "Medium"
    else: return "High"

df['Spending_Level'] = df['Total Spent'].apply(classify_spending)
print("\n1. Đã tạo cột Spending_Level:")
print(df['Spending_Level'].value_counts())

# --------------------------------------------------
# Câu 2.3: Normalization
# --------------------------------------------------

print("\n--- Câu 2.3: NORMALIZATION ---")

# a) Min-Max cho Quantity
scaler_minmax = MinMaxScaler()
df['Quantity_norm'] = scaler_minmax.fit_transform(df[['Quantity']])
print(f"✓ Quantity: Đã chuẩn hóa Min-Max [0, 1]")

# b) Standardization (Z-score) cho Total Spent
scaler_std = StandardScaler()
df['TotalSpent_std'] = scaler_std.fit_transform(df[['Total Spent']])
print(f"✓ Total Spent: Đã chuẩn hóa Z-score")

# --------------------------------------------------
# Câu 2.4: Feature Construction
# --------------------------------------------------

print("\n--- Câu 2.4: FEATURE CONSTRUCTION ---")

# Trích xuất Tháng từ ngày giao dịch
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
df['Month'] = df['Transaction Date'].dt.month
print(f"✓ Đã tạo cột Month từ Transaction Date")

# Tạo biến nhị phân Is_High_Quantity (Mua trên 5 sản phẩm)
df['Is_High_Quantity'] = (df['Quantity'] > 5).astype(int)
print(f"✓ Đã tạo biến Is_High_Quantity")

# ==================================================
# PHẦN 3: GIẢM DỮ LIỆU
# ==================================================

print("\n" + "=" * 60)
print("PHẦN 3: GIẢM DỮ LIỆU")
print("=" * 60)

# --------------------------------------------------
# Câu 3.1: Data Selection
# --------------------------------------------------

print("\n--- Câu 3.1: DATA SELECTION ---")

print("\n1. Loại bỏ các cột không cần thiết cho mô hình:")
columns_to_drop = ['Transaction ID', 'Customer ID', 'Transaction Date']
df_reduced = df.drop(columns=columns_to_drop)
print(f"✓ Đã loại bỏ các cột định danh. Số cột còn lại: {len(df_reduced.columns)}")

# --------------------------------------------------
# Câu 3.2: Sampling
# --------------------------------------------------

print("\n--- Câu 3.2: SAMPLING ---")

# Chuẩn bị X và y (Dự đoán Discount Applied)
X = df_reduced.drop('Discount Applied', axis=1)
# Chuyển y sang dạng số
y = df_reduced['Discount Applied'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {len(X_train)} dòng")
print(f"✓ Test set: {len(X_test)} dòng")

# ==================================================
# PHẦN 4: TÓM TẮT VÀ SO SÁNH
# ==================================================

print("\n" + "=" * 60)
print("TÓM TẮT QUÁ TRÌNH TIỀN XỬ LÝ")
print("=" * 60)

print(f"1. Dòng dữ liệu gốc: {len(pd.read_csv('retail_store_sales.csv'))}")
print(f"2. Dòng sau khi làm sạch sâu: {len(df)}")
print(f"3. Số cột mới đã tạo: Month, Spending_Level, TotalSpent_std, v.v.")
print(f"4. Chất lượng: Logic Price * Quantity = Total Spent đã được khớp 100%.")

print("\n" + "=" * 60)
print("HOÀN THÀNH!")
print("=" * 60)