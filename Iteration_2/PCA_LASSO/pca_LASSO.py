
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

# Configuration settings
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load and preprocess data"""
    feature_path = Path(r"D:\Desktop\Pca_LASSO_\final_40x10.xlsx")
    target_path = Path(r"D:\Desktop\Pca_LASSO_\Index.xlsx")
    
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    
    X = pd.read_excel(feature_path, header=None).values
    y = pd.read_excel(target_path).iloc[:, 0].values
    
    assert len(X) == len(y), f"Data size mismatch: Features {len(X)} vs Target {len(y)}"
    return X, y

def evaluate_model(y_true, y_pred, model_name):
    """Model evaluation"""
    return {
        "Model": model_name,
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
    }

def plot_results(y_true, predictions, model_names):
    """Visualization comparison"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for y_pred, name in zip(predictions, model_names):
        plt.scatter(y_true, y_pred, alpha=0.6, label=name)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for y_pred, name in zip(predictions, model_names):
        residuals = y_true - y_pred
        plt.hist(residuals, bins=15, alpha=0.5, label=name)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

def main():
    # 1. Data preparation
    X, y = load_data()
    print(f"Data shape: X{X.shape}, y{y.shape}")
    
    # 2. Data splitting & standardization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model definition & training
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
        best_model = None  # Explicit initialization
        
        # Special handling for PCR
        if name == "PCR":
            pca = config["pca"]
            model = config["model"]
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            best_model = model  # Critical fix
        else:
            if isinstance(config, GridSearchCV):
                config.fit(X_train_scaled, y_train)
                best_model = config.best_estimator_
            else:
                best_model = config.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
        
        predictions.append(y_pred)
        results.append(evaluate_model(y_test, y_pred, name))
        
        # Cross-validation
        cv_scores = cross_val_score(
            best_model, X, y, cv=5, scoring='r2'
        )
        print(f"\n{name} Cross-validation R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

    # 4. Results presentation
    result_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(result_df)
    plot_results(y_test, predictions, models.keys())
    result_df.to_excel("model_comparison_results.xlsx", index=False)

if __name__ == "__main__":
    main()

