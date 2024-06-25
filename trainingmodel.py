from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd


# Load the data
data_path = '/cleaned_train_data.csv'  # replace with your data path
data = pd.read_csv(data_path)

# Define X and y
X = data.drop(columns=['price'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = HistGradientBoostingRegressor()


# Define the parameter distribution
param_distributions = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300, 500],
    'max_leaf_nodes': [10, 20, 31, 50,63, 127, 255, 1000],
    'max_depth': [3, 5, 7, 9,10],
    'min_samples_leaf': [1, 5, 10, 20, 50, 100],
    'l2_regularization': [0, 0.1, 0.5, 1, 5, 10],
}

# Setup the randomized search with parallelization
random_search = RandomizedSearchCV(
    HistGradientBoostingRegressor(),
    param_distributions,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

# Fit the random search
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the results
print("Training Metrics:")
print(f"MSE: {mse_train}")
print(f"RMSE: {rmse_train}")
print(f"MAE: {mae_train}")
print(f"R²: {r2_train}")

print("\nValidation Metrics:")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")

# Save the model
model_path = '/best_model.pkl'