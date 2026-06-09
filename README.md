````markdown name=README.md
# traffic-flow-prediction

Traffic Flow Prediction using **Linear Regression** (Traditional Machine Learning).

## 项目说明

本项目使用**线性回归（Linear Regression）**方法完成交通流量预测，支持从 `.mat` 文件加载数据，完成预处理、训练、评估与可视化。

### 实现功能

- ✅ **数据加载**：支持 `tra_X_tr`, `tra_X_te`, `tra_Y_tr`, `tra_Y_te`, `tra_adj_mat`
- ✅ **数据预处理**：
  - 缺失值填充（中位数策略）
  - IQR 异常值裁剪（1.5倍IQR）
  - 标准化（StandardScaler）
  - **特征增强**（考虑空间相关性）
  - PCA 降维
- ✅ **线性回归模型**：`sklearn.linear_model.LinearRegression`
- ✅ **模型评估**：MAE、RMSE、MAPE、R²、交叉验证 RMSE
- ✅ **可视化（5种）**：
  - 实际值 vs 预测值散点图
  - 模型性能指标柱状图
  - 时间序列预测对比（第一个位置）
  - 残差分布直方图
  - 残差 vs 预测值散点图

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py --mat-path /path/to/traffic_dataset.mat --output-dir outputs
```

### 可选参数

- `--pca-components`：PCA维度（默认：64）
- `--cv`：交叉验证折数（默认：3）

### 示例

```bash
python main.py \
    --mat-path D:\Admin\Downloads\traffic_dataset.mat \
    --output-dir outputs \
    --pca-components 64 \
    --cv 3
```

## 输出结果

程序会在输出目录中生成以下文件：

| 文件名 | 说明 |
|------|------|
| `model_performance_report.csv` | 模型性能指标（CSV格式） |
| `model_performance_report.json` | 模型性能指标（JSON格式） |
| `actual_vs_predicted.png` | 实际值 vs 预测值散点图 |
| `model_comparison.png` | 性能指标柱状图 |
| `time_series_prediction.png` | 时间序列预测对比（第一个位置） |
| `residual_distribution.png` | 残差分布直方图 |
| `residuals_vs_predictions.png` | 残差 vs 预测值散点图 |

## 项目结构

```
traffic-flow-prediction/
├── traffic_flow_prediction/
│   ├── __init__.py              # 包初始化
│   ├── data_loader.py           # 数据加载模块
│   ├── preprocessing.py         # 数据预处理模块
│   ├── models.py                # 模型定义
│   ├── evaluation.py            # 模型评估
│   ├── visualization.py         # 可视化模块
│   └── pipeline.py              # 完整工作流
├── main.py                      # 命令行接口
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

## 技术栈

- **Python 3.x**
- **scikit-learn**：机器学习库
- **numpy, pandas**：数据处理
- **matplotlib, seaborn**：数据可视化
- **scipy**：加载.mat文件

## 数据格式

输入 `.mat` 文件需包含以下变量：

| 变量名 | 形状 | 说明 |
|-------|------|------|
| `tra_X_tr` | 1×1261 | 训练集输入（1261个时间步） |
| `tra_X_te` | 1×840 | 测试集输入（840个时间步） |
| `tra_Y_tr` | 36×1261 | 训练集输出（36个位置×1261个时间步） |
| `tra_Y_te` | 36×840 | 测试集输出（36个位置×840个时间步） |
| `tra_adj_mat` | 36×36 | 邻接矩阵（位置间空间连接性） |

每个 `tra_X_*` 元素为 36×48 矩阵（36个位置×48个特征）

## 特征工程

### 原始特征
- 36个地点×48个特征 = 1728个特征

### 增强特征
1. **原始特征**：1728维
2. **位置平均**：36维（每个位置的48个特征平均值）
3. **位置标准差**：36维（每个位置的48个特征标准差）
4. **全局平均**：36维（所有地点每个特征的平均值）
5. **邻接特征**：36维（使用邻接矩阵计算的空间上下文）

**总计**：1728 + 36 + 36 + 36 + 36 = **1872个特征**

### 预处理步骤
1. 中位数填补缺失值
2. IQR方法裁剪异常值
3. StandardScaler标准化
4. **PCA降维到64维**（可配置）

## 模型参数

### Linear Regression
```python
LinearRegression()
```
- 使用 scikit-learn 默认参数
- 拟合方法：最小二乘法（OLS）
- 支持多输出回归（36个位置的流量同时预测）

## 评估指标

| 指标 | 公式 | 范围 | 说明 |
|------|------|------|------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | [0, ∞) | 平均绝对误差，越小越好 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | [0, ∞) | 均方根误差，越小越好 |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | [0, ∞) | 平均绝对百分比误差(%)，越小越好 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | (-∞, 1] | 决定系数，越接近1越好 |
| **CV RMSE** | 3折交叉验证的RMSE均值 | [0, ∞) | 模型稳定性指标，越小越好 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
# 将你的 traffic_dataset.mat 放在合适的位置

# 3. 运行程序
python main.py --mat-path ./traffic_dataset.mat --output-dir ./outputs

# 4. 查看结果
# 输出文件会保存在 outputs/ 目录
```

## 注意事项

- 确保 `.mat` 文件包含所有必需的变量
- PCA 维度不应超过训练样本数量
- 交叉验证折数 `cv` 不应超过样本数量
- 对于大型数据集，可能需要较长的计算时间

## 许可证

MIT License
````
