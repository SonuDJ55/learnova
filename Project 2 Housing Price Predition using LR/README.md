# Residential Housing Price Prediction Engine

An end-to-end, production-grade machine learning pipeline designed to predict residential housing prices with precision. This project addresses classic structural challenges in real-world tabular data—such as right-skewed targets, multi-category text features, data leakage across validation folds, and feature scale disparities—by utilizing robust preprocessing techniques and an iterative gradient descent optimization framework.

---

## 🛠️ Technical Architecture & Stack

The core architecture isolates data transformations within a strict, leak-proof cross-validation workflow before passing clean vectors to an online iterative estimator.

### 1. Data Engineering & Preprocessing Layer
* **Pandas & NumPy**: Utilized for vectorization, exploratory statistical profiling, structural cleaning, and executing non-linear target transformations.
* **Feature-Engine (`Winsorizer`)**: Configured natively using the Interquartile Range (IQR) method (`capping_method="iqr"`, `fold=1.5`, `tail="both"`) to clip extreme continuous feature outliers dynamically within validation splits.
* **Scikit-Learn Preprocessing & Composite Engines**:
    * `StandardScaler`: Standardizes continuous and discrete features to achieve zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$), ensuring smooth convergence gradients.
    * `OneHotEncoder`: Encodes nominal text vectors while implementing `drop="first"` to actively eliminate dummy variable traps and curb artificial multicollinearity (VIF inflation).
    * `ColumnTransformer`: Serves as the centralized feature orchestration switchboard, segregating pipelines by data type to maintain raw alignment across data records.

### 2. Modeling & Optimization Layer
* **Scikit-Learn `SGDRegressor`**: Implements an iterative, stochastic gradient descent linear regression model. Chosen for its performance efficiency and structural flexibility.
* **Regularization**: Integrates native **L2 Regularization (Ridge Penalty)** directly into the gradient update step (`penalty="l2"`) to constrain regression coefficient magnitudes ($\beta$) and structurally lower model variance.
* **Learning Rate Schedule**: Configured with an `adaptive` descent profile (`learning_rate="adaptive"`, `eta0=0.01`), dynamically scaling step sizes downwards whenever the optimization loss plateaus.

---

## 📈 Key Data Engineering Insights

### 🛡️ Preventing Cross-Validation Data Leakage
A critical vulnerability in machine learning pipelines is applying feature tracking or capping (like Winsorization or scaling) globally across an entire dataset prior to splitting. This allows statistical profiles ($\mu$, $\sigma$, or percentile fences) from validation data to leak directly into the training context, leading to overly optimistic cross-validation metrics. 

This project solves leakage by sealing the `Winsorizer` and `StandardScaler` inside a sequential `Pipeline` nested within a `ColumnTransformer`. Thresholds are strictly computed on active training splits and applied downstream to out-of-fold validation sets.

### 🎯 Overcoming Target Right-Skewness via Log Transformations
During baseline evaluations, the model displayed a structural performance gap between data splits:
* **Training $R^2$ Score**: ~`0.6936`
* **Testing $R^2$ Score**: ~`0.6347`