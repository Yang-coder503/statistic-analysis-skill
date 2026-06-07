# 模型评估与调优

## 目录
1. [回归模型评估](#回归模型评估)
2. [分类模型评估](#分类模型评估)
3. [聚类评估](#聚类评估)
4. [交叉验证策略](#交叉验证策略)
5. [超参数调优](#超参数调优)
6. [模型诊断](#模型诊断)
7. [模型选择与对比](#模型选择与对比)

---

## 回归模型评估

### 指标速查

| 指标 | 公式本质 | 特点 | 何时用 |
|------|---------|------|--------|
| **MSE** | 误差平方的均值 | 对大误差惩罚重 | 大误差不可接受时 |
| **RMSE** | sqrt(MSE) | 与目标同单位，最常用 | 默认首选 |
| **MAE** | 误差绝对值的均值 | 对异常值不敏感 | 异常值较多时 |
| **R²** | 解释的方差比例 | 0~1，越接近1越好 | 需要可解释性 |
| **MAPE** | 百分比误差均值 | 相对误差，直观 | 报告给非技术人员 |
| **Explained Variance** | 类似 R² | 检测系统性偏差 | 诊断用 |

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
    explained_variance_score
)
import numpy as np

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")   # 越小越好，与 y 同单位
print(f"MAE:  {mae:.4f}")   # 越小越好
print(f"R²:   {r2:.4f}")    # 越接近 1 越好
print(f"MAPE: {mape:.2%}")  # 越小越好

# 评估：RMSE 和 MAE 差距大 → 存在一些大误差 → 检查异常值
if rmse > mae * 1.5:
    print("⚠️ RMSE >> MAE，存在较大的预测误差，建议检查残差分布")
```

---

## 分类模型评估

### 指标选择指南

| 场景 | 关注指标 | 原因 |
|------|---------|------|
| 欺诈检测 | Recall（召回率） | 漏掉欺诈的代价远大于误报 |
| 垃圾邮件过滤 | Precision（精确率） | 把正常邮件标为垃圾比漏标垃圾更糟 |
| 医学筛查 | Recall | 宁可误诊也不能漏诊 |
| 精准广告投放 | Precision | 不要浪费广告费在不感兴趣的人身上 |
| 类别平衡的通用评估 | F1-Score / AUC | 综合考量 |
| 类别严重不平衡 | AUC / PR-AUC | Accuracy 在类别不平衡时会误导 |

### 完整代码

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 正类概率

# === 基础指标 ===
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:    {average_precision_score(y_test, y_prob):.4f}")

# === 详细报告 ===
print(classification_report(y_test, y_pred, target_names=['负类', '正类']))

# === 混淆矩阵可视化 ===
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测负', '预测正'],
            yticklabels=['实际负', '实际正'], ax=ax)
ax.set_ylabel('实际'); ax.set_xlabel('预测')

# === ROC + PR 双图 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].fill_between(fpr, tpr, alpha=0.2)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC 曲线'); axes[0].legend()

# Precision-Recall（类别不平衡时更有信息量）
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec, prec, label=f'AP = {average_precision_score(y_test, y_prob):.3f}')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall 曲线'); axes[1].legend()
```

### 阈值调优

```python
# 默认阈值 0.5 不一定最佳，可以找最优阈值
from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 1.0, 0.05)
scores = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    scores.append({
        'threshold': t,
        'precision': precision_score(y_test, y_pred_t),
        'recall': recall_score(y_test, y_pred_t),
        'f1': f1_score(y_test, y_pred_t)
    })

scores_df = pd.DataFrame(scores)
best = scores_df.loc[scores_df['f1'].idxmax()]
print(f"最优阈值: {best['threshold']:.2f} (F1={best['f1']:.3f})")

# 查看所有阈值
fig, ax = plt.subplots(figsize=(10, 5))
scores_df.set_index('threshold')[['precision', 'recall', 'f1']].plot(ax=ax)
ax.axvline(best['threshold'], color='red', linestyle='--')
ax.set_xlabel('阈值'); ax.set_ylabel('分数')
```

---

## 聚类评估

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

labels = km.labels_
X = X_scaled

sil = silhouette_score(X, labels)    # [-1, 1]，越高越好
dbi = davies_bouldin_score(X, labels)  # 越小越好
ch = calinski_harabasz_score(X, labels) # 越高越好

print(f"轮廓系数 (Silhouette):      {sil:.4f}")
print(f"Davies-Bouldin Index:        {dbi:.4f}")
print(f"Calinski-Harabasz Index:     {ch:.4f}")
```

---

## 交叉验证策略

### 选择正确的 CV 策略

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
)

# 1. 分类问题 → StratifiedKFold（保持每折类别比例一致）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. 回归问题 → KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. 时间序列 → TimeSeriesSplit（绝不能打乱顺序！）
cv = TimeSeriesSplit(n_splits=5)

# 4. 有分组结构（如用户级数据） → GroupKFold
# cv = GroupKFold(n_splits=5)

# 使用
from sklearn.model_selection import cross_val_score, cross_validate

scores = cross_validate(
    model, X, y,
    cv=cv,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
    return_train_score=True
)
print(f"Test R²: {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")
print(f"Train R²: {scores['train_r2'].mean():.4f}")
# 如果 Train R² >> Test R² → 过拟合
# 如果两者都低 → 欠拟合
```

### 学习曲线（诊断过拟合/欠拟合）

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='训练集')
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='验证集')
ax.fill_between(train_sizes,
                test_scores.mean(axis=1) - test_scores.std(axis=1),
                test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.2)
ax.set_xlabel('训练样本数'); ax.set_ylabel('R²')
ax.legend(); ax.set_title('学习曲线')
# 解读：
# - 两条线差距大 → 过拟合 → 加正则化/减特征/加数据
# - 两条线都低 → 欠拟合 → 加特征/换更复杂模型
```

---

## 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# === GridSearchCV（小搜索空间，穷举） ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,       # 并行计算
    verbose=1
)
grid.fit(X_train, y_train)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳 AUC: {grid.best_score_:.4f}")

# 查看所有结果
cv_results = pd.DataFrame(grid.cv_results_)
top5 = cv_results.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print(top5)

# === RandomizedSearchCV（大搜索空间，随机采样，推荐） ===
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,     # 只试 50 组，不穷举
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

---

## 模型诊断

### 过拟合 vs 欠拟合

| 现象 | 训练集表现 | 验证集表现 | 解决方案 |
|------|-----------|-----------|---------|
| 欠拟合 | 差 | 差 | 更复杂模型、更多特征、减少正则化 |
| 刚好 | 好 | 好 | ✅ 保持 |
| 过拟合 | 很好 | 差 | 正则化、减少特征、更多数据、早停、Dropout |

### 残差诊断（回归）

```python
residuals = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 残差 vs 拟合值
axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('拟合值'); axes[0, 0].set_ylabel('残差')
axes[0, 0].set_title('残差 vs 拟合值')

# 2. Q-Q plot
from scipy import stats
stats.probplot(residuals, dist='norm', plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot（检查正态性）')

# 3. 残差直方图
axes[1, 0].hist(residuals, bins=30, edgecolor='white')
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('残差'); axes[1, 0].set_title('残差分布')

# 4. 实际 vs 预测
axes[1, 1].scatter(y_test, y_pred, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1, 1].set_xlabel('实际值'); axes[1, 1].set_ylabel('预测值')
axes[1, 1].set_title('实际 vs 预测（完美预测应落在对角线上）')

plt.tight_layout()
```

---

## 模型选择与对比

```python
# 全流程对比
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
}

results = []
for name, model in models.items():
    # 交叉验证
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=['roc_auc', 'accuracy', 'f1'],
        return_train_score=True
    )
    results.append({
        '模型': name,
        'AUC': f"{cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}",
        'Accuracy': f"{cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}",
        'F1': f"{cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}",
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```
