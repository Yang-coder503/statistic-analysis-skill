# 数据预处理与特征工程

## 目录
1. [缺失值处理](#缺失值处理)
2. [异常值检测与处理](#异常值检测与处理)
3. [数据类型转换](#数据类型转换)
4. [特征编码](#特征编码)
5. [特征缩放](#特征缩放)
6. [特征工程](#特征工程)
7. [数据划分](#数据划分)

---

## 缺失值处理

### 第一步：诊断

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno  # pip install missingno

# 缺失值概览
missing = pd.DataFrame({
    '缺失数': df.isnull().sum(),
    '缺失比例(%)': (df.isnull().mean() * 100).round(2),
    '数据类型': df.dtypes
}).query('缺失数 > 0').sort_values('缺失比例(%)', ascending=False)
print(missing)

# 可视化缺失模式
msno.matrix(df)      # 缺失值矩阵图
msno.heatmap(df)     # 缺失值相关性（某些列倾向于同时缺失？）
msno.dendrogram(df)  # 层次聚类缺失模式
```

### 第二步：决定策略

```python
# 决策树：如何处理缺失值
# ├── 缺失 > 50% → 考虑删除整列（信息太少）
# ├── 缺失 30-50% → 慎用填充，考虑把"是否缺失"作为特征
# ├── 缺失 5-30% → 根据数据类型选择填充策略
# └── 缺失 < 5% → 简单填充或删除行即可
```

### 第三步：执行

```python
# === 删除 ===
df.dropna(subset=['critical_col'], inplace=True)  # 仅对关键列
df.dropna(thresh=len(df.columns) * 0.5, inplace=True)  # 保留至少50%非空的行

# === 填充 ===
# 数值列
df['col'].fillna(df['col'].median(), inplace=True)        # 中位数，偏态分布
df['col'].fillna(df['col'].mean(), inplace=True)           # 均值，正态分布
df['col'].fillna(df['col'].mode()[0], inplace=True)        # 众数
df['col'].fillna(method='ffill', inplace=True)             # 前向填充（时序）
df['col'].fillna(method='bfill', inplace=True)             # 后向填充
df['col'].interpolate(method='linear', inplace=True)       # 线性插值（时序）

# 分组填充（更精确）
df['col'] = df.groupby('category')['col'].transform(lambda x: x.fillna(x.median()))

# 分类列
df['cat_col'].fillna('未知', inplace=True)  # 新类别

# === 标记缺失 ===
# "是否缺失"本身可能携带信息！
df['col_missing'] = df['col'].isnull().astype(int)
```

---

## 异常值检测与处理

### 检测方法

```python
# === 方法1：IQR 法（推荐，不依赖正态假设） ===
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
print(f"IQR 异常值: {len(outliers)} 个 ({len(outliers)/len(df)*100:.1f}%)")

# === 方法2：Z-score 法（适合正态分布，>3 视为异常） ===
from scipy.stats import zscore
z = np.abs(zscore(df['col'].dropna()))
outliers_z = df['col'].dropna()[z > 3]

# === 方法3：百分位法 ===
lower_pct = df['col'].quantile(0.01)  # 低于1%分位
upper_pct = df['col'].quantile(0.99)  # 高于99%分位

# === 可视化辅助 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x=df['col'], ax=axes[0])
sns.histplot(df['col'], bins=50, ax=axes[1])
```

### 处理策略

```python
# 1. 缩尾处理（Winsorize，保留数据量）
df['col_winsorized'] = df['col'].clip(lower=lower, upper=upper)

# 2. 对数变换（压缩右偏分布的大值）
df['col_log'] = np.log1p(df['col'])  # log(1+x)，处理 0 值

# 3. 删除（仅当确认是数据错误）
df = df[(df['col'] >= lower) & (df['col'] <= upper)]

# 4. 用替代值替换
df.loc[(df['col'] < lower) | (df['col'] > upper), 'col'] = df['col'].median()

# ⚠️ 注意：对于价格、收入等自然偏态变量，"异常值"可能是正常的！
# 不要机械地删除 — 始终先理解业务含义
```

---

## 数据类型转换

```python
# 数值转换（强制转换，非法值变 NaN）
df['col'] = pd.to_numeric(df['col'], errors='coerce')

# 日期转换
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')

# Category 类型（节省内存，适合重复值少的字符串列）
# 如果某列的唯一个数 < 行数的 50%，很适合转 category
for col in df.select_dtypes('object').columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype('category')

# 布尔列
df['flag'] = df['col'].astype(bool)
```

---

## 特征编码

```python
# === One-Hot 编码（无序分类变量） ===
df_encoded = pd.get_dummies(df, columns=['gender', 'city'], drop_first=True)
# drop_first=True 避免完全多重共线性

# === Label Encoding（有序分类变量） ===
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# 有序编码（适用于有序分类如 低<中<高）
ord_enc = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['level_encoded'] = ord_enc.fit_transform(df[['level']])

# === 目标编码（高基数分类变量，如邮编、产品ID）===
# 对每个类别，用该类别目标均值来编码（需在交叉验证内做避免泄漏）
# 使用 category_encoders 库
# import category_encoders as ce
# encoder = ce.TargetEncoder(cols=['zip_code'])
# df['zip_encoded'] = encoder.fit_transform(df['zip_code'], df['target'])

# === 频率编码（无目标变量时的替代方案）===
freq_map = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq_map)
```

---

## 特征缩放

**什么时候必须做缩放？** 使用基于距离或梯度的模型时：
- **必须缩放**：KNN、SVM、K-Means、PCA、逻辑回归、神经网络
- **不需要**：决策树、随机森林、XGBoost、朴素贝叶斯
- **可做可不做**：线性回归（不做影响系数可比较性，但不影响预测）

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler：均值0，标准差1（默认首选）
# 适合：数据大致正态，大多数算法
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# MinMaxScaler：缩放到 [0, 1] 区间
# 适合：需要非负值（如图像像素）、保留稀疏结构
scaler = MinMaxScaler()
X_mm = scaler.fit_transform(X)

# RobustScaler：基于中位数和 IQR
# 适合：数据有很多异常值，不想被它们影响
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)

# ⚠️ 务必在 train_test_split 之后用 train 拟合 scaler，再 transform test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled = scaler.transform(X_test)          # 只用 transform！
```

---

## 特征工程

### 数值特征

```python
# 多项式特征（捕获非线性关系）
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['x1', 'x2']])

# 分箱（将连续值离散化，捕获非线性趋势）
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100],
                        labels=['少年', '青年', '壮年', '中年', '老年'])

# 对数/平方根变换（压缩偏态分布）
df['amount_log'] = np.log1p(df['amount'])
df['area_sqrt'] = np.sqrt(df['area'])

# 交互特征
df['price_per_unit'] = df['total_price'] / (df['quantity'] + 1e-9)
df['area_ratio'] = df['living_area'] / (df['total_area'] + 1e-9)
```

### 时间特征

```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
df['hour'] = df['date'].dt.hour  # 如果有精确时间

# 周期性编码（避免把 23点 和 0点 当成"差23"）
df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
```

### 聚合特征（从关联表提取信息）

```python
# 用户级别的聚合特征
user_features = df.groupby('user_id').agg(
    total_orders=('order_id', 'count'),
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    first_order=('date', 'min'),
    last_order=('date', 'max'),
    unique_categories=('category', 'nunique'),
).reset_index()

user_features['active_days'] = (user_features['last_order'] - user_features['first_order']).dt.days
```

---

## 数据划分

```python
from sklearn.model_selection import train_test_split

# 标准划分（分类问题用 stratify 确保比例一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 分类问题加 stratify
)

# 时间序列：绝不能随机打乱！用时间切分
split_date = '2024-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
```
