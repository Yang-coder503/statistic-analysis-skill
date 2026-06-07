# Pandas 常用操作速查

## 目录
1. [数据读写](#数据读写)
2. [数据查看与信息](#数据查看与信息)
3. [列操作](#列操作)
4. [行筛选 (Filter)](#行筛选)
5. [排序](#排序)
6. [分组聚合](#分组聚合)
7. [连接与合并](#连接与合并)
8. [重塑数据](#重塑数据)
9. [窗口函数与滚动计算](#窗口函数)
10. [字符串操作](#字符串操作)
11. [日期时间操作](#日期时间操作)
12. [性能技巧](#性能技巧)

---

## 数据读写

```python
import pandas as pd

# CSV
df = pd.read_csv('file.csv', encoding='utf-8')
df = pd.read_csv('file.csv', sep='\t', header=0)       # TSV 文件
df = pd.read_csv('file.csv', parse_dates=['date_col'])  # 自动解析日期
df.to_csv('output.csv', index=False, encoding='utf-8-sig')  # Excel 兼容中文

# Excel（需要 openpyxl 或 xlrd）
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df_dict = pd.read_excel('file.xlsx', sheet_name=None)   # 读取所有 sheet，返回 dict
df.to_excel('output.xlsx', sheet_name='结果', index=False)

# JSON
df = pd.read_json('file.json')
df = pd.json_normalize(json_data, record_path='records', meta=['id'])

# Parquet / Feather（高速读写，推荐大数据使用）
df = pd.read_parquet('file.parquet')
df = pd.read_feather('file.feather')

# 直接从剪贴板（快速实验）
df = pd.read_clipboard()
```

## 数据查看与信息

```python
df.shape          # (行数, 列数)
df.columns.tolist()  # 列名列表
df.dtypes         # 每列数据类型
df.info()         # 内存使用、类型、非空计数
df.describe()     # 数值列汇总统计
df.describe(include='object')  # 分类列汇总
df.nunique()      # 每列唯一值数量
df.memory_usage(deep=True)  # 每列内存占用
```

## 列操作

```python
# 选择列
df['col']                  # 单列 → Series
df[['col1', 'col2']]       # 多列 → DataFrame
df.filter(like='keyword')  # 列名包含关键词的
df.filter(regex=r'^col_')  # 正则匹配列名

# 重命名
df.rename(columns={'old': 'new'}, inplace=True)
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 新增列
df['new_col'] = df['a'] + df['b']
df['new_col'] = df.apply(lambda row: row['a'] * 2 if row['b'] > 0 else 0, axis=1)
df.assign(new_col = lambda x: x['a'] / x['b'])  # 函数式写法

# 删除列
df.drop(['col1', 'col2'], axis=1, inplace=True)

# 类型转换
df['col'] = pd.to_numeric(df['col'], errors='coerce')  # 转数值，非法值变 NaN
df['col'] = df['col'].astype('category')  # 转分类类型（节省内存）
```

## 行筛选

```python
# 布尔索引
df[df['age'] > 18]
df[(df['age'] > 18) & (df['city'] == '北京')]  # 多条件用 & | ~（注意括号！）

# .loc（按标签）与 .iloc（按位置）
df.loc[5:10, ['name', 'age']]     # 标签切片（包含右端点）
df.iloc[5:10, [0, 2]]             # 位置切片（不包含右端点）

# .query() 方法（可读性更好）
df.query('age > 18 and city == "北京"')
df.query('age.between(20, 30)')   # 区间筛选

# 常用筛选模式
df[df['name'].str.contains('张')]           # 字符串包含
df[df['date'].between('2024-01', '2024-06')] # 日期区间
df[df['col'].isin(['A', 'B', 'C'])]          # 值在列表中
df[df['col'].notna()]                         # 非空值
df.nlargest(10, 'score')                      # 前10大
df.nsmallest(10, 'score')                     # 前10小

# 按类型筛选
df.select_dtypes(include=['number'])          # 只选数值列
df.select_dtypes(include=['object', 'category'])
```

## 排序

```python
df.sort_values('col', ascending=False)              # 单列排序
df.sort_values(['col1', 'col2'], ascending=[False, True])  # 多列
df.sort_index()                                      # 按索引排序
```

## 分组聚合

```python
# 基础分组
df.groupby('category')['value'].mean()
df.groupby('category').agg({'col1': 'mean', 'col2': 'sum'})

# 多重聚合
df.groupby('category').agg(
    mean_val=('value', 'mean'),
    sum_val=('value', 'sum'),
    count_val=('value', 'count'),
    std_val=('value', 'std'),
)

# 自定义聚合函数
df.groupby('category').agg(lambda x: x.quantile(0.9) - x.quantile(0.1))  # 十分位距

# transform（保持原形状，用于特征工程）
df['group_mean'] = df.groupby('category')['value'].transform('mean')
df['pct_rank'] = df.groupby('category')['value'].transform(lambda x: x.rank(pct=True))

# filter（按组筛选）
df.groupby('category').filter(lambda x: len(x) > 10)  # 保留样本量>10的组
```

## 连接与合并

```python
# 纵向拼接
pd.concat([df1, df2], ignore_index=True)  # 像 SQL UNION ALL

# 横向拼接
pd.concat([df1, df2], axis=1)

# 表连接（JOIN）
pd.merge(left, right, on='key', how='inner')       # SQL INNER JOIN
pd.merge(left, right, on='key', how='left')        # SQL LEFT JOIN
pd.merge(left, right, left_on='key1', right_on='key2', how='outer')
```

## 重塑数据

```python
# 宽表 → 长表（melt，类似 Excel 逆透视）
df_long = df.melt(
    id_vars=['id', 'name'],        # 保持不变的列
    value_vars=['q1', 'q2', 'q3'], # 要转换的列
    var_name='quarter',             # 新列名：原列名
    value_name='sales'              # 新列名：原值
)

# 长表 → 宽表（pivot）
df_wide = df.pivot(index='date', columns='category', values='sales')

# 数据透视表（带聚合）
df.pivot_table(
    index='date',
    columns='category',
    values='sales',
    aggfunc='sum',
    fill_value=0,
    margins=True          # 添加汇总行列
)

# 交叉表（频次统计）
pd.crosstab(df['gender'], df['purchased'], normalize='index')
```

## 窗口函数

```python
# 滚动窗口（移动平均）
df['ma7'] = df['value'].rolling(window=7).mean()

# 扩展窗口（累计统计）
df['cum_mean'] = df['value'].expanding().mean()

# 指数加权
df['ewm'] = df['value'].ewm(span=7, adjust=False).mean()

# 排名（类 SQL RANK / ROW_NUMBER）
df['rank'] = df.groupby('group')['value'].rank(method='dense', ascending=False)

# 差分（lag/lead）
df['diff_1'] = df['value'].diff(1)        # 一阶差分
df['pct_change'] = df['value'].pct_change()  # 环比变化率
df['shift_1'] = df['value'].shift(1)      # 上一行（lag）
df['shift_-1'] = df['value'].shift(-1)    # 下一行（lead）
```

## 字符串操作

```python
# 通过 .str 访问器
df['name'].str.upper()
df['name'].str.strip()
df['name'].str.replace('旧', '新', regex=False)
df['name'].str.contains('关键词', na=False)
df['name'].str.extract(r'(\d{4})')          # 提取正则匹配的第一组
df['name'].str.extractall(r'(\w+)')         # 提取所有匹配
df['name'].str.split('_', expand=True)      # 拆分为多列
```

## 日期时间操作

```python
# 确保 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 通过 .dt 访问器
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['dayofweek'] = df['date'].dt.dayofweek     # 0=周一
df['weekday_name'] = df['date'].dt.day_name()  # 'Monday'
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# 日期算术
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days

# 时间重采样（resample）
df.set_index('date').resample('M')['value'].sum()     # 月汇总
df.set_index('date').resample('W-MON')['value'].mean() # 周汇总（周一为起始）
```

## 性能技巧

1. **使用合适的 dtype**：`category` 类型对重复值多的列可节省大量内存
2. **向量化操作优于 apply**：`df['a'] + df['b']` 比 `df.apply(...)` 快 10-100 倍
3. **query() 优于布尔索引**：大数据集上 `query()` 有优化
4. **分块读取大文件**：`pd.read_csv('big.csv', chunksize=100000)`
5. **使用 Parquet 格式**：比 CSV 快 5-10 倍，且保留数据类型
