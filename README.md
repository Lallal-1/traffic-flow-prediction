# traffic-flow-prediction

Traffic Flow Prediction using Traditional Machine Learning.

## 项目说明

本项目使用传统机器学习方法完成交通流量预测，支持从 `.mat` 文件加载数据，完成预处理、训练、评估与可视化。

### 已实现功能

- 数据加载：支持 `tra_X_tr`, `tra_X_te`, `tra_Y_tr`, `tra_Y_te`, `tra_adj_mat`
- 数据预处理：缺失值填充、IQR异常值裁剪、标准化、PCA降维、空间邻接特征增强
- 传统模型：
  - Linear Regression
  - Ridge Regression
  - Decision Tree Regressor
  - SVM Regression (LinearSVR + MultiOutputRegressor)
  - Random Forest
- 模型评估：MAE、RMSE、MAPE、R²、交叉验证 RMSE
- 可视化（至少5种）：
  - 实际值 vs 预测值
  - 模型性能对比
  - 时间序列预测对比
  - 残差分布
  - 残差 vs 预测值
  - 特征重要性分析

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py --mat-path /absolute/path/to/your_dataset.mat --output-dir outputs
```

可选参数：

- `--pca-components`：PCA维度（默认64）
- `--cv`：交叉验证折数（默认3）

## 输出结果

程序会在输出目录中生成：

- `model_performance_report.csv`
- `model_performance_report.json`
- `actual_vs_predicted.png`
- `model_comparison.png`
- `time_series_prediction.png`
- `residual_distribution.png`
- `residuals_vs_predictions.png`
- `feature_importance.png`（若可用）
