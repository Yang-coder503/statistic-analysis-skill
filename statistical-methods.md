# 统计与机器学习方法知识库

## 目录
1. [描述性统计](#1-描述性统计)
2. [概率分布与检验](#2-概率分布与检验)
3. [假设检验](#3-假设检验)
4. [相关分析](#4-相关分析)
5. [回归分析](#5-回归分析)
6. [分类方法](#6-分类方法)
7. [聚类分析](#7-聚类分析)
8. [降维与特征选择](#8-降维与特征选择)
9. [时间序列分析](#9-时间序列分析)
10. [集成学习](#10-集成学习)
11. [方法选择决策树](#11-方法选择决策树)

---

## 1. 描述性统计

### 核心概念

描述性统计是数据分析的起点，回答"数据长什么样"。

**集中趋势度量：**
- **均值 (mean)**：对异常值敏感，适合大致对称的分布
- **中位数 (median)**：对异常值鲁棒，适合偏态分布（如收入数据）
- **众数 (mode)**：适合分类数据和发现重复值

**离散程度度量：**
- **标准差 (std)**：与均值同单位，68-95-99.7 规则（正态假设）
- **方差 (var)**：标准差的平方，用于数学推导
- **IQR (四分位距)**：Q3 - Q1，用于箱线图和异常值检测
- **极差 (range)**：max - min，极粗略

**分布形状：**
- **偏度 (skewness)**：>0 右偏（长尾在右），<0 左偏
- **峰度 (kurtosis)**：>3 厚尾（比正态更多极端值）

```python
import pandas as pd
import numpy as np
from scipy import stats

# 全面的描述性统计
desc = df.describe(include='all').T
desc['skewness'] = df.select_dtypes(include=[np.number]).apply(stats.skew)
desc['kurtosis'] = df.select_dtypes(include=[np.number]).apply(stats.kurtosis)
desc['missing'] = df.isnull().sum()
desc['missing_pct'] = df.isnull().mean() * 100

# 分类变量的频率表
freq = df['category'].value_counts()
freq_pct = df['category'].value_counts(normalize=True)
```

### 何时用哪个指标
- 数据有极端异常值 → 用中位数 + IQR 而非均值 + 标准差
- 汇报"平均水平" → 正态数据用均值，偏态数据用中位数
- 比较不同单位的变异程度 → 用变异系数 CV = std/mean

---

## 2. 概率分布与检验

### 常见分布

| 分布 | 描述 | 典型场景 |
|------|------|---------|
| **正态分布** | 钟形对称 | 测量误差、身高体重 |
| **二项分布** | n 次试验成功次数 | 转化率、通过率 |
| **泊松分布** | 单位时间内事件数 | 网站访问、投诉数 |
| **指数分布** | 等待时间 | 设备故障间隔 |
| **均匀分布** | 等可能区间 | 随机抽样 |

### 正态性检验

```python
from scipy.stats import shapiro, normaltest, kstest

# Shapiro-Wilk（推荐，适用于 n < 5000）
stat, p = shapiro(df['value'])
print(f"Shapiro-Wilk: p={p:.4f} → {'正态' if p > 0.05 else '非正态'}")

# D'Agostino K²（适用于大样本）
stat, p = normaltest(df['value'])

# Q-Q 图（视觉检查，比统计检验更实用）
import scipy.stats as stats
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
stats.probplot(df['value'], dist='norm', plot=ax)
```

> **重要**：大样本时（n > 5000），统计检验几乎总是拒绝正态假设，此时应依靠 Q-Q 图和偏度/峰度进行视觉判断。

---

## 3. 假设检验

### 检验方法选择表

| 场景 | 检验方法 | 非参数替代 |
|------|---------|-----------|
| 单样本均值 vs 已知值 | 单样本 t 检验 | Wilcoxon 符号秩 |
| 两组独立样本均值 | 独立样本 t 检验 | Mann-Whitney U |
| 两组配对样本均值 | 配对 t 检验 | Wilcoxon 符号秩 |
| 三组及以上均值 | 单因素 ANOVA | Kruskal-Wallis H |
| 多因素均值 | 双因素/多因素 ANOVA | — |
| 分类变量独立性 | 卡方检验 | Fisher 精确检验 |
| 方差齐性 | Levene / Bartlett | — |

### Python 实现

```python
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency
from scipy.stats import mannwhitneyu, kruskal, levene

# 独立样本 t 检验
group_a = df[df['group'] == 'A']['score']
group_b = df[df['group'] == 'B']['score']

# 先检验方差齐性
stat_levene, p_levene = levene(group_a, group_b)
equal_var = p_levene > 0.05  # p>0.05 认为方差齐

# 执行 t 检验
stat, p = ttest_ind(group_a, group_b, equal_var=equal_var)
print(f"t={stat:.3f}, p={p:.4f} → {'显著' if p < 0.05 else '不显著'}")

# 效应量（Cohen's d，评估差异的实际大小）
cohens_d = (group_a.mean() - group_b.mean()) / np.sqrt(
    (group_a.var() + group_b.var()) / 2
)
print(f"Cohen's d = {cohens_d:.3f}")  # 0.2=小, 0.5=中, 0.8=大

# 单因素 ANOVA
groups = [df[df['category'] == c]['value'] for c in df['category'].unique()]
stat, p = f_oneway(*groups)

# 如果 ANOVA 显著，做事后检验（两两比较）
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['value'], df['category'], alpha=0.05)
print(tukey)

# 卡方检验（分类变量关联）
contingency = pd.crosstab(df['gender'], df['purchased'])
chi2, p, dof, expected = chi2_contingency(contingency)
```

### 解读关键点
- **p < 0.05** 表示「在零假设成立的条件下，观察到当前数据或更极端数据的概率小于 5%」— 不是「零假设成立的概率」
- **p 值不衡量效应大小**——p 值很小只是说差异不太可能是随机产生的，不说明差异有多大。始终配合效应量（Cohen's d、η²、Cramér's V）一起报告
- **大样本下 p 值容易显著**——n > 10,000 时微小的实际差异也可能显著，此时重点看效应量

---

## 4. 相关分析

```python
# Pearson（线性相关，要求正态性）
# Spearman（秩相关，对非线性单调关系也有效，更鲁棒）
# Kendall（秩相关，对异常值最鲁棒）

corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')

# 带显著性检验
from scipy.stats import pearsonr, spearmanr
r, p = pearsonr(df['x'], df['y'])
print(f"Pearson r={r:.3f}, p={p:.4f}")
```

**注意**：相关性 ≠ 因果性。存在第三变量（混杂因子）可能导致伪相关。

---

## 5. 回归分析

### 方法选择

| 场景 | 方法 | 何时用 |
|------|------|--------|
| 线性可分的连续预测 | 线性回归 | 基线模型，可解释性强 |
| 多重共线性严重 | Ridge 回归 | L2 正则化收缩系数 |
| 需要特征选择 | Lasso 回归 | L1 正则化把不重要系数压为 0 |
| 两者兼顾 | ElasticNet | 混合 L1 + L2 |
| 非线性关系 | 多项式回归 | 添加 x², x³ 等项 |
| 强非线性 | 决策树/随机森林回归 | 自动发现交互和非线性 |
| 二分类结果 | 逻辑回归 | 输出概率，可解释性强 |
| 计数数据 | 泊松回归 | 结果是非负整数 |

### 线性回归完整模板

```python
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# === 1. 准备数据 ===
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# 标准化（Lasso/Ridge 必须做，对系数比较也有帮助）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 2. Statsmodels 版本（详细统计推断，适合论文） ===
X_sm = sm.add_constant(X_train)  # 添加截距项
model_sm = sm.OLS(y_train, X_sm).fit()
print(model_sm.summary())
# 关注：R², Adj R², 每个系数的 p 值, F 统计量, AIC/BIC

# === 3. Sklearn 版本（预测导向） ===
model = LinearRegression()
model.fit(X_train, y_train)

# 系数解读
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df)

# 交叉验证评估
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"CV R²: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")

# === 4. 正则化回归（多重共线性或特征多时） ===
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge R²: {ridge.score(X_test, y_test):.3f}")

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
# 检查哪些特征系数被压缩为 0
selected = pd.Series(lasso.coef_, index=X.columns)
print(f"Lasso 选出了 {(lasso.coef_ != 0).sum()} 个特征")

# === 5. 残差诊断 ===
y_pred = model.predict(X_test)
residuals = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# 残差 vs 拟合值（检查线性/等方差）
axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('预测值'); axes[0, 0].set_ylabel('残差')
axes[0, 0].set_title('残差 vs 预测值（检查线性/等方差）')
# Q-Q plot（检查正态性）
stats.probplot(residuals, dist='norm', plot=axes[0, 1])
# 残差直方图
axes[1, 0].hist(residuals, bins=30, edgecolor='white')
# 残差 vs 各特征
axes[1, 1].scatter(X_test.iloc[:, 0], residuals, alpha=0.5)
axes[1, 1].set_xlabel(X.columns[0])
plt.tight_layout()
```

### 逻辑回归

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 二分类
lr = LogisticRegression(penalty='l2', C=1.0, random_state=42)
lr.fit(X_train, y_train)

y_prob = lr.predict_proba(X_test)[:, 1]  # 正类概率
y_pred = lr.predict(X_test)

print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(classification_report(y_test, y_pred))

# 查看系数含义（对数几率变化）
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0],
    'odds_ratio': np.exp(lr.coef_[0])  # e^系数 = 优势比
})
```

---

## 6. 分类方法

### 方法比较

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **逻辑回归** | 可解释性强，输出概率 | 假设线性决策边界 | 基线模型，需要解释 |
| **KNN** | 简单直观，无训练 | 预测慢，维度灾难 | 小数据，特征少的基线 |
| **SVM** | 高维有效，核技巧灵活 | 大样本慢，调参敏感 | 文本分类，高维数据 |
| **决策树** | 可解释，自动特征选择 | 容易过拟合 | 需要解释规则的场景 |
| **随机森林** | 鲁棒，自动处理非线性 | 可解释性差，模型大 | 通用分类/回归 |
| **XGBoost/LightGBM** | 竞赛级精度 | 调参复杂 | 追求最高精度 |
| **朴素贝叶斯** | 极快，适合高维稀疏 | 特征独立假设不现实 | 文本分类，实时预测 |

### 通用分类流程

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 基线模型
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, eval_metric='logloss'),
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"{name}: AUC = {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
```

---

## 7. 聚类分析

### 方法选择

| 方法 | 何时用 | 关键参数 |
|------|--------|---------|
| **K-Means** | 簇数已知，球形簇，量级相当 | k（簇数） |
| **DBSCAN** | 簇数未知，任意形状，含噪声 | eps, min_samples |
| **层次聚类** | 需要展示簇间层级，小数据 | linkage 方法 |
| **高斯混合** | 软聚类，每个点属于多个簇 | n_components |

### K-Means + 肘法则 + 轮廓系数

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 标准化（聚类对量纲敏感！）
X_scaled = StandardScaler().fit_transform(df[numeric_features])

# 肘法则找最佳 k
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('K'); axes[0].set_ylabel('簇内平方和 (Inertia)')
axes[0].set_title('肘法则 (Elbow Method)')
axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_xlabel('K'); axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数 (越高越好)')

# 最终模型
best_k = K_range[np.argmax(silhouettes)]
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(X_scaled)

# 各簇特征画像
cluster_profile = df.groupby('cluster')[numeric_features].mean()
print(cluster_profile)
```

### 降维可视化（聚类结果用 PCA/t-SNE 展示）

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA 可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='Set2', alpha=0.6)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, label='Cluster')
```

---

## 8. 降维与特征选择

### PCA

```python
from sklearn.decomposition import PCA

pca = PCA().fit(X_scaled)

# 累计方差解释率
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1  # 保留 95% 方差
print(f"保留 {n_components} 个主成分即可达到 95% 方差解释率")

# 可视化
fig, ax = plt.subplots()
ax.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
ax.axhline(0.95, color='red', linestyle='--', label='95%')
ax.axvline(n_components, color='green', linestyle='--')
ax.set_xlabel('主成分数'); ax.set_ylabel('累计方差解释率')
ax.legend()
```

### 特征重要性

```python
# 基于随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance['feature'][-15:], importance['importance'][-15:])
ax.set_title('Top 15 重要特征')
```

---

## 9. 时间序列分析

### 关键步骤
1. **平稳性检验** (ADF test)
2. **ACF/PACF 图** — 确定 AR 和 MA 的阶数
3. **模型拟合** — ARIMA/SARIMA
4. **残差诊断** — 残差应为白噪声
5. **预测**

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# 平稳性检验
result = adfuller(df['value'].dropna())
print(f"ADF 统计量={result[0]:.3f}, p={result[1]:.4f}")
if result[1] > 0.05:
    print("→ 序列不平稳，需要差分！")
    df['diff'] = df['value'].diff()

# ACF / PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df['value'].dropna(), lags=40, ax=axes[0])
plot_pacf(df['value'].dropna(), lags=40, ax=axes[1])

# ARIMA 手动模式
model = ARIMA(df['value'], order=(p, d, q))  # 从 ACF/PACF 确定 p, d, q
fitted = model.fit()
print(fitted.summary())

# 自动选择（pmdarima）
# from pmdarima import auto_arima
# auto_model = auto_arima(df['value'], seasonal=True, m=12, trace=True)

# 预测
forecast = fitted.forecast(steps=12)  # 预测未来 12 期
```

---

## 10. 集成学习

| 方法 | Bagging（如随机森林） | Boosting（如 XGBoost） |
|------|---------------------|------------------------|
| 核心思路 | 并行训练多模型，投票/平均 | 串行训练，后一个修正前一个的错误 |
| 抗过拟合 | 很好 | 需要正则化和早停 |
| 训练速度 | 快（可并行） | 慢（串行） |
| 代表方法 | RandomForest, ExtraTrees | XGBoost, LightGBM, CatBoost, AdaBoost |

```python
# XGBoost 基线
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
```

---

## 11. 方法选择决策树

```
目标是什么？
├── 预测连续值（回归）
│   ├── 线性关系 → 线性回归 / Ridge / Lasso
│   ├── 非线性关系
│   │   ├── 可接受黑箱 → 随机森林 / XGBoost
│   │   └── 需要可解释 → 决策树回归 / 多项式回归
│   └── 时间序列 → ARIMA / Prophet
│
├── 预测分类（分类）
│   ├── 需要概率 + 可解释 → 逻辑回归
│   ├── 少量特征、大量样本 → SVM / 逻辑回归
│   ├── 复杂非线性、精度优先 → XGBoost / LightGBM
│   └── 文本/图像 → 朴素贝叶斯 / 深度学习
│
├── 发现分组（聚类）
│   ├── 已知簇数、球形 → K-Means
│   ├── 未知簇数、含噪声 → DBSCAN
│   └── 想看层级关系 → 层次聚类
│
├── 降维（压缩）
│   ├── 保留全局结构 → PCA
│   ├── 保留局部结构（可视化） → t-SNE
│   └── 大规模数据可视化 → UMAP
│
├── 比较组间差异 → t检验 / ANOVA / 卡方检验
├── 探索变量关系 → 相关分析 / 散点图
└── 筛选重要特征 → 互信息 / Lasso / RF 重要性
```

> 经验法则：**从简单模型开始（线性回归/逻辑回归），建立了性能基线后再尝试复杂模型。** 如果线性模型已经足够好，就没有理由引入更复杂的模型。
