# SQL 分析与 Pandas 联动

## 目录
1. [连接方式总览](#连接方式总览)
2. [SQLite 本地分析](#sqlite-本地分析)
3. [MySQL 连接](#mysql-连接)
4. [PostgreSQL 连接](#postgresql-连接)
5. [SQLAlchemy 统一接口](#sqlalchemy-统一接口)
6. [常用 SQL 分析模式](#常用-sql-分析模式)
7. [Pandas + SQL 混合分析技巧](#pandas--sql-混合分析技巧)
8. [大查询优化](#大查询优化)

---

## 连接方式总览

| 数据库 | 推荐驱动 | 连接字符串格式 |
|--------|---------|---------------|
| SQLite | sqlite3 (内置) | `sqlite:///path/to/db.sqlite` |
| MySQL | pymysql | `mysql+pymysql://user:pass@host:port/db` |
| PostgreSQL | psycopg2 | `postgresql://user:pass@host:port/db` |

> **建议**：使用 SQLAlchemy 的 `create_engine()` 统一管理连接，方便在不同数据库之间切换。

---

## SQLite 本地分析

SQLite 是课程项目和本地分析最常用的选择 — 无需安装服务端，数据库就是一个单文件。

### 基础连接

```python
import sqlite3
import pandas as pd

# 方式一：sqlite3 直连
conn = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM users LIMIT 10", conn)
conn.close()

# 方式二：SQLAlchemy（推荐，功能更强）
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.sqlite')
df = pd.read_sql_query("SELECT * FROM users", engine)
```

### 快速创建内存数据库（实验/测试）

```python
engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine, index=False, if_exists='replace')
# 现在可以对这个内存表执行 SQL 查询了
```

### CSV 文件直接当作 SQL 查询

```python
# 把 CSV 读入 SQLite，然后用 SQL 分析
engine = create_engine('sqlite:///analysis.db')
df_csv = pd.read_csv('sales.csv')
df_csv.to_sql('sales', engine, index=False, if_exists='replace')

# 现在可以用 SQL 了
result = pd.read_sql_query("""
    SELECT region, SUM(revenue) as total
    FROM sales
    WHERE year = 2024
    GROUP BY region
    ORDER BY total DESC
""", engine)
```

---

## MySQL 连接

### 安装依赖

```bash
pip install pymysql sqlalchemy
```

### 连接代码

```python
from sqlalchemy import create_engine
import pandas as pd

# 连接配置
DB_CONFIG = {
    'host': 'localhost',      # 或远程 IP
    'port': 3306,
    'user': 'your_username',
    'password': 'your_password',
    'database': 'your_db',
    'charset': 'utf8mb4',     # 支持中文和 emoji
}

# 构建连接字符串
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    f"?charset={DB_CONFIG['charset']}"
)

# 测试连接
with engine.connect() as conn:
    result = conn.execute("SELECT VERSION()").fetchone()
    print(f"MySQL 版本: {result[0]}")

# 查询数据
df = pd.read_sql_query("SELECT * FROM orders WHERE amount > 100", engine)
```

### 安全提示

**永远不要在代码中硬编码密码。** 推荐使用环境变量：

```python
import os
password = os.environ.get('MYSQL_PASSWORD')  # 运行前设置：export MYSQL_PASSWORD=xxx
```

---

## PostgreSQL 连接

### 安装依赖

```bash
pip install psycopg2-binary sqlalchemy
```

### 连接代码

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    'postgresql://user:password@localhost:5432/database'
)

# PostgreSQL 特有的 schema 支持
df = pd.read_sql_query(
    "SELECT * FROM analytics.user_events WHERE event_date >= '2024-01-01'",
    engine
)
```

---

## SQLAlchemy 统一接口

使用 SQLAlchemy 后，你可以用同一套代码操作不同数据库，只需要改连接字符串：

```python
from sqlalchemy import create_engine, text
import pandas as pd

# 根据数据库类型切换
def get_engine(db_type='sqlite', **kwargs):
    if db_type == 'sqlite':
        return create_engine(f"sqlite:///{kwargs['path']}")
    elif db_type == 'mysql':
        return create_engine(
            f"mysql+pymysql://{kwargs['user']}:{kwargs['password']}"
            f"@{kwargs['host']}:{kwargs.get('port', 3306)}/{kwargs['database']}"
        )
    elif db_type == 'postgresql':
        return create_engine(
            f"postgresql://{kwargs['user']}:{kwargs['password']}"
            f"@{kwargs['host']}:{kwargs.get('port', 5432)}/{kwargs['database']}"
        )
    raise ValueError(f"Unsupported db_type: {db_type}")

engine = get_engine('mysql', user='root', password='xxx', host='localhost', database='mydb')

# 通用查询函数
def query(sql, engine):
    """执行 SQL 并返回 DataFrame"""
    return pd.read_sql_query(sql, engine)

# 写入数据库
def to_db(df, table_name, engine, if_exists='replace'):
    """将 DataFrame 写入数据库"""
    df.to_sql(table_name, engine, index=False, if_exists=if_exists)
```

---

## 常用 SQL 分析模式

### 基础聚合分析

```sql
-- 按类别汇总
SELECT
    category,
    COUNT(*) as cnt,
    AVG(price) as avg_price,
    SUM(revenue) as total_revenue,
    MIN(date) as first_date,
    MAX(date) as last_date
FROM orders
WHERE date >= '2024-01-01'
GROUP BY category
HAVING COUNT(*) >= 10          -- 只保留样本量足够的类别
ORDER BY total_revenue DESC;
```

### 窗口函数分析

```sql
-- 每月销售额和环比增长
SELECT
    year_month,
    monthly_sales,
    LAG(monthly_sales) OVER (ORDER BY year_month) as prev_month_sales,
    ROUND(
        (monthly_sales - LAG(monthly_sales) OVER (ORDER BY year_month))
        / LAG(monthly_sales) OVER (ORDER BY year_month) * 100, 2
    ) as mom_growth_pct,
    AVG(monthly_sales) OVER (
        ORDER BY year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as ma3  -- 3个月移动平均
FROM monthly_summary;
```

### 用户行为漏斗分析

```sql
WITH funnel AS (
    SELECT
        user_id,
        MAX(CASE WHEN event = 'visit' THEN 1 ELSE 0 END) as visited,
        MAX(CASE WHEN event = 'register' THEN 1 ELSE 0 END) as registered,
        MAX(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) as purchased
    FROM user_events
    WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
    GROUP BY user_id
)
SELECT
    SUM(visited) as visits,
    SUM(registered) as registrations,
    SUM(purchased) as purchases,
    ROUND(SUM(registered) * 100.0 / SUM(visited), 2) as reg_rate,
    ROUND(SUM(purchased) * 100.0 / SUM(registered), 2) as purchase_rate
FROM funnel;
```

### RFM 分析（客户价值分层）

```sql
SELECT
    customer_id,
    JULIANDAY('now') - JULIANDAY(MAX(order_date)) as recency_days,
    COUNT(*) as frequency,
    SUM(amount) as monetary,
    NTILE(4) OVER (ORDER BY JULIANDAY('now') - JULIANDAY(MAX(order_date)) DESC) as r_score,
    NTILE(4) OVER (ORDER BY COUNT(*)) as f_score,
    NTILE(4) OVER (ORDER BY SUM(amount)) as m_score
FROM orders
GROUP BY customer_id;
```

---

## Pandas + SQL 混合分析技巧

### 技巧1：SQL 粗筛 → Pandas 精细分析

SQL 负责从百万级数据中高效过滤出几千行，Pandas 负责灵活的统计分析和可视化：

```python
# Step 1: SQL 过滤和聚合（速度快）
df_raw = pd.read_sql_query("""
    SELECT customer_id, order_date, amount, category
    FROM orders
    WHERE order_date >= '2023-01-01'
""", engine)

# Step 2: Pandas 灵活分析
df_raw['month'] = pd.to_datetime(df_raw['order_date']).dt.to_period('M')
pivot = df_raw.pivot_table(
    index='month', columns='category', values='amount', aggfunc='sum'
)
```

### 技巧2：Pandas 结果回写到 SQL

```python
# 分析完成后，将结果写回数据库
summary = df.groupby('category').agg(...).reset_index()
summary.to_sql('category_summary', engine, index=False, if_exists='replace')

# 然后在 SQL 中做更多 JOIN 分析
final = pd.read_sql_query("""
    SELECT c.*, s.total_revenue
    FROM categories c
    LEFT JOIN category_summary s ON c.id = s.category_id
""", engine)
```

### 技巧3：用 Pandas 生成 SQL

```python
# 当需要批量查询时，用 Pandas 生成 SQL IN 列表
ids = df['customer_id'].unique()
ids_str = ','.join([f"'{id}'" for id in ids[:1000]])  # 注意 SQL 长度限制

sql = f"""
    SELECT * FROM orders
    WHERE customer_id IN ({ids_str})
"""
result = pd.read_sql_query(sql, engine)
```

---

## 大查询优化

1. **用 LIMIT 测试**：先跑 `LIMIT 10` 确认查询逻辑正确，再跑全量
2. **只选需要的列**：`SELECT col1, col2` 而不是 `SELECT *`
3. **WHERE 优先过滤**：在 JOIN 之前尽量缩小数据范围
4. **使用索引列过滤**：对经常查询的列建议创建索引
5. **分块读取**：
   ```python
   # 每次读 50000 行，避免内存溢出
   for chunk in pd.read_sql_query(sql, engine, chunksize=50000):
       processed = process(chunk)
       processed.to_csv('output.csv', mode='a', header=False)
   ```
6. **Chunked 读 CSV + 写 SQL**：
   ```python
   for chunk in pd.read_csv('large.csv', chunksize=50000):
       chunk.to_sql('table', engine, index=False, if_exists='append')
   ```
