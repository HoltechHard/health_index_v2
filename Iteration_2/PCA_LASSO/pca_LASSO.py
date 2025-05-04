"""
模型对比脚本（修复UnboundLocalError）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

# 配置参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据和预处理"""
    feature_path = Path(r"D:\Desktop\Pca_LASSO_\final_40x10.xlsx")
    target_path = Path(r"D:\Desktop\Pca_LASSO_\Index.xlsx")
    
    if not feature_path.exists():
        raise FileNotFoundError(f"特征文件不存在：{feature_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"目标文件不存在：{target_path}")
    
    X = pd.read_excel(feature_path, header=None).values
    y = pd.read_excel(target_path).iloc[:, 0].values
    
    assert len(X) == len(y), f"数据量不匹配: 特征{len(X)}条，目标{len(y)}条"
    return X, y

def evaluate_model(y_true, y_pred, model_name):
    """模型评估"""
    return {
        "Model": model_name,
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
    }

def plot_results(y_true, predictions, model_names):
    """可视化对比"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for y_pred, name in zip(predictions, model_names):
        plt.scatter(y_true, y_pred, alpha=0.6, label=name)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for y_pred, name in zip(predictions, model_names):
        residuals = y_true - y_pred
        plt.hist(residuals, bins=15, alpha=0.5, label=name)
    plt.xlabel("残差")
    plt.ylabel("频数")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

def main():
    # 1. 数据准备
    X, y = load_data()
    print(f"数据形状：X{X.shape}, y{y.shape}")
    
    # 2. 数据拆分与标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 模型定义与训练
    models = {
        "PCR": {
            "pca": PCA(n_components=0.95),
            "model": LinearRegression()
        },
        "LASSO": GridSearchCV(
            Lasso(max_iter=10000),
            {"alpha": [0.001, 0.01, 0.1, 1, 10]},
            cv=5
        ),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = []
    predictions = []
    
    for name, config in models.items():
        best_model = None  # 显式初始化变量
        
        # PCR模型特殊处理
        if name == "PCR":
            # PCA转换
            pca = config["pca"]
            model = config["model"]
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            # 训练模型
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            best_model = model  # 关键修复点
        else:
            # 其他模型训练
            if isinstance(config, GridSearchCV):
                config.fit(X_train_scaled, y_train)
                best_model = config.best_estimator_
            else:
                best_model = config.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
        
        # 记录结果
        predictions.append(y_pred)
        results.append(evaluate_model(y_test, y_pred, name))
        
        # 交叉验证
        cv_scores = cross_val_score(
            best_model, X, y, cv=5, scoring='r2'
        )
        print(f"\n{name}模型交叉验证R²：{np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

    # 4. 结果展示与保存
    result_df = pd.DataFrame(results)
    print("\n模型性能对比：")
    print(result_df)
    plot_results(y_test, predictions, models.keys())
    result_df.to_excel("model_comparison_results.xlsx", index=False)

if __name__ == "__main__":
    main()

