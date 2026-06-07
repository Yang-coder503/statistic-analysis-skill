# 数据可视化指南

## 目录
1. [图表选择速查（极简）](#图表选择速查)
2. [库选择：何时用什么](#库选择)
3. [Matplotlib + Seaborn 模板（出版级静态图）](#matplotlib--seaborn-模板)
4. [Plotly 模板（交互式探索图）](#plotly-模板)
5. [常用图表代码模板](#常用图表代码模板)
6. [学术报告美化清单](#学术报告美化清单)
7. [中文显示配置](#中文显示配置)

---

## 图表选择速查

| 分析目的 | 推荐图表 |
|---------|---------|
| 单变量分布 | 直方图 `histplot`、密度图 `kdeplot`、箱线图 `boxplot` |
| 类别计数 | 柱状图 `barplot`、计数图 `countplot` |
| 两连续变量关系 | 散点图 `scatterplot`、hexbin 图 |
| 连续 vs 类别 | 分组箱线图、小提琴图 `violinplot`、蜂群图 `swarmplot` |
| 相关性矩阵 | 热力图 `heatmap` |
| 时间序列趋势 | 折线图 `lineplot`、面积图 |
| 比例/占比 | 饼图（慎用）、堆叠柱状图、树图（plotly） |
| 多维比较 | 平行坐标图、雷达图、散点矩阵 `pairplot` |
| 排名对比 | 横向柱状图（sorted） |
| 模型评估 | ROC 曲线、混淆矩阵热力图、残差图、学习曲线 |

---

## 库选择

| 场景 | 推荐库 | 原因 |
|------|--------|------|
| 课程报告、论文 | Matplotlib + Seaborn | 出版级静态图，DPI 可控，中文支持好 |
| Jupyter 探索分析 | Plotly | 可交互，支持缩放和悬停，看数据方便 |
| PPT/汇报展示 | Plotly 或 Seaborn | Plotly 交互性强，Seaborn 美观默认风格好 |
| 大量数据（>10万点） | Matplotlib（rasterized=True） | Plotly 在大数据量下性能差 |
| 地理数据 | Plotly (mapbox) 或 folium | 交互地图 |

---

## Matplotlib + Seaborn 模板

### 全局配置

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体（见底部详细配置）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# plt.rcParams['font.sans-serif'] = ['SimHei']           # Windows
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Seaborn 风格
sns.set_style('whitegrid')  # 或 'darkgrid', 'white', 'ticks'
sns.set_context('notebook') # 或 'paper', 'talk', 'poster'
sns.set_palette('Set2')     # 推荐色板：Set2, husl, muted, Blues
```

### 通用画图模板

```python
# --- 创建画布 ---
fig, ax = plt.subplots(figsize=(10, 6))

# --- 绘图 ---
# （在这里放具体的绘图代码）

# --- 标题和标签 ---
ax.set_title('图表标题', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('X 轴标签', fontsize=12)
ax.set_ylabel('Y 轴标签', fontsize=12)

# --- 图例 ---
ax.legend(loc='best', frameon=True, fontsize=10)

# --- 网格和边框 ---
ax.spines['top'].set_visible(False)    # 隐藏上、右边框（更简洁）
ax.spines['right'].set_visible(False)

# --- 保存 ---
plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```

---

## Plotly 模板

### 全局配置

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 默认主题
import plotly.io as pio
pio.templates.default = 'plotly_white'  # 简洁白底

# 中文标题在 Plotly 中直接写中文即可，一般不需要额外配置
```

### 通用画图模板

```python
fig = px.scatter(
    df, x='x_col', y='y_col',
    color='category',           # 按类别着色
    size='size_col',            # （可选）气泡大小
    hover_data=['col1', 'col2'], # 悬停时显示的额外信息
    title='图表标题',
    labels={'x_col': 'X 轴标签', 'y_col': 'Y 轴标签'},
    trendline='ols'             # （可选）添加趋势线，或 'lowess'
)

fig.update_layout(
    width=800, height=600,
    title_font_size=18,
    hovermode='x unified',      # 统一悬停框
)

# 保存为 HTML（交互式）或静态图片
fig.write_html('interactive_plot.html')
fig.write_image('static_plot.png', scale=2)  # 需要 kaleido: pip install kaleido
```

---

## 常用图表代码模板

### 1. 分布图（直方图 + 密度曲线）

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='value', bins=30, kde=True,
             color='steelblue', edgecolor='white', alpha=0.7, ax=ax)
ax.set_title('数值分布', fontsize=14, fontweight='bold')
# 添加统计线
ax.axvline(df['value'].mean(), color='red', linestyle='--', label=f"均值={df['value'].mean():.2f}")
ax.axvline(df['value'].median(), color='orange', linestyle='-', label=f"中位数={df['value'].median():.2f}")
ax.legend()
```

### 2. 分组箱线图 / 小提琴图

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 箱线图：适合展示四分位数和异常值
sns.boxplot(data=df, x='category', y='value', palette='Set2', ax=axes[0])
axes[0].set_title('分组箱线图')

# 小提琴图：展示完整分布形状
sns.violinplot(data=df, x='category', y='value', palette='Set2',
               inner='quartile', ax=axes[1])  # inner='box'/'quartile'/'stick'
axes[1].set_title('分组小提琴图')
```

### 3. 相关性热力图

```python
corr_matrix = df.select_dtypes(include=[np.number]).corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下半三角
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            mask=mask, vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': '相关系数'},
            ax=ax)
ax.set_title('特征相关性矩阵', fontsize=16, fontweight='bold')
```

### 4. 分类柱状图（含排序和数值标签）

```python
# 先聚合 + 排序
summary = df.groupby('category')['value'].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(summary.index, summary.values, color=sns.color_palette('Blues_d', len(summary)))

# 数值标签
for bar, val in zip(bars, summary.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=10)

ax.set_title('各类别平均值对比', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### 5. 散点图 + 回归线

```python
g = sns.jointplot(
    data=df, x='x_col', y='y_col',
    kind='reg',           # 回归线; 也可用 'hex', 'kde'
    height=8,
    scatter_kws={'alpha': 0.3, 's': 10},
    line_kws={'color': 'red'}
)
g.fig.suptitle('两变量关系', y=1.02)
# 同时显示 Pearson 相关系数
r, p = df['x_col'].corr(df['y_col']), 0  # scipy.stats.pearsonr 获取 p 值
g.ax_joint.text(0.05, 0.95, f'r = {r:.3f}', transform=g.ax_joint.transAxes, fontsize=12)
```

### 6. 散点矩阵（多变量探索）

```python
# Seaborn 版本
g = sns.pairplot(
    df,
    vars=['col1', 'col2', 'col3', 'col4'],  # 选几列
    hue='category',      # 按类别着色
    diag_kind='kde',     # 对角线上画 KDE
    plot_kws={'alpha': 0.5, 's': 15},
    palette='Set2'
)

# Plotly 版本（交互式）
fig = px.scatter_matrix(
    df,
    dimensions=['col1', 'col2', 'col3', 'col4'],
    color='category',
    opacity=0.5
)
```

### 7. 时间序列折线图

```python
fig, ax = plt.subplots(figsize=(14, 6))

# 多组时间序列
for name, group in df.groupby('category'):
    ts = group.set_index('date')['value'].resample('M').mean()
    ax.plot(ts.index, ts.values, label=name, linewidth=2, alpha=0.8)

ax.set_title('月度趋势', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', frameon=True)
ax.set_xlabel('')
ax.set_ylabel('数值')

# Plotly 版本
fig = px.line(df, x='date', y='value', color='category',
              title='时间序列趋势', markers=False)
```

### 8. ROC 曲线（模型评估）

```python
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 8))

for name, y_true, y_score in models:  # models = [(name, y_true, y_pred_proba), ...]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测', alpha=0.5)
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假阳性率 (FPR)'); ax.set_ylabel('真阳性率 (TPR)')
ax.set_title('ROC 曲线对比', fontweight='bold')
ax.legend(loc='lower right')
ax.set_aspect('equal')
```

### 9. 多子图布局

```python
# 规整的网格布局
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # 把 2x3 展平成一维数组

for i, col in enumerate(numeric_cols[:6]):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette('Set2')[i % 8])
    axes[i].set_title(f'{col} 分布')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--')

# 隐藏多余的子图（如果不够6个）
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
```

### 10. Plotly 仪表板式布局

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('销售趋势', '类别占比', 'Top 10 产品', '散点分析'),
    specs=[[{'type': 'xy'}, {'type': 'domain'}],     # domain 用于饼图
           [{'type': 'xy'}, {'type': 'xy'}]]
)

fig.add_trace(go.Scatter(x=df['date'], y=df['sales'], mode='lines'), row=1, col=1)
fig.add_trace(go.Pie(labels=df['category'], values=df['count']), row=1, col=2)
# ... 更多子图

fig.update_layout(height=800, title_text='数据分析仪表板')
fig.show()
```

---

## 学术报告美化清单

> 输出图表用于课程论文或报告时，逐项检查：

- [ ] **DPI ≥ 300**（`dpi=300` 或 `scale=2`）
- [ ] **字体大小 ≥ 10pt**（`fontsize` 参数，确保打印清晰）
- [ ] **中英文混排正确**（中文用中文字体，数字用英文字体）
- [ ] **颜色对色盲友好**（推荐 `cmap='viridis'` 或 Seaborn 色板）
- [ ] **坐标轴有标签和单位**
- [ ] **图例放在不遮挡数据的位置**
- [ ] **去掉多余的边框**（`spines` 设置）
- [ ] **保存为矢量格式（PDF/SVG）以便排版**：`plt.savefig('fig.pdf', format='pdf')`
- [ ] **如果图片在 LaTeX 中使用，考虑 `pgf` 后端**

---

## 中文显示配置

```python
# === 完整的中文配置代码（按操作系统选择） ===
import matplotlib.pyplot as plt
import matplotlib

# 方法1：系统字体（推荐）
import platform
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC']
elif system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 方法2：使用 font_manager 指定字体文件
# from matplotlib.font_manager import FontProperties
# chinese_font = FontProperties(fname='/path/to/font.ttf', size=12)

# 检测可用字体（找中文字体）
# import matplotlib.font_manager as fm
# chinese_fonts = [f.name for f in fm.fontManager.ttflist if any(
#     kw in f.name.lower() for kw in ['cjk', 'hei', 'song', 'ming', 'kai', 'chinese']
# )]
# print(chinese_fonts)
```
