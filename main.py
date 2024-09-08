from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import itertools

def process_data():
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    x = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # drop null values and duplicates and line up features and targets
    x_clean = x.dropna()
    y_clean = y.loc[x_clean.index]
    x_clean = x_clean.drop_duplicates()
    y_clean = y_clean.loc[x_clean.index]

    # encode diagnosis to either 0 or 1, indicating benign or malignant tumor
    label_encoder = LabelEncoder()
    y_encode = label_encoder.fit_transform(y_clean.values.ravel())
    return x_clean, y_encode

def standardize(x_clean):
    scaler = StandardScaler()
    return scaler.fit_transform(x_clean)

# 80/20 training/test split
def split(x_clean, y_encode):
    return train_test_split(x_clean, y_encode, random_state=104, test_size=0.2, shuffle=True)

# trains data using SGDRegressor, with different iterations, tolerances, and learning rates
def train(x_train, y_train, max_iter, tol, learning_rate):
    sgd_regressor = SGDRegressor(max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    sgd_regressor.fit(x_train, y_train)
    y_pred = sgd_regressor.predict(x_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    return sgd_regressor, r2, mse

def hyperparameter_tuning(x_train, y_train):
    # define ranges for hyperparameters
    max_iter_values = [1000, 3000, 5000]
    tol_values = [1e-4, 1e-3, 1e-2]
    learning_rate_values = ['constant', 'optimal', 'invscaling']

    # gets all combos of parameters for testing
    param_combinations = itertools.product(max_iter_values, tol_values, learning_rate_values)

    best_model = None
    best_mse = float('inf')

    # iterate through all combinations and log results
    for max_iter, tol, learning_rate in param_combinations:
        model, r2, mse = train(x_train, y_train, max_iter, tol, learning_rate)
        params = f"max_iter={max_iter}, tol={tol}, learning_rate={learning_rate}"
        log_results(params, mse, r2)
        # picks model based on best mse value    
        if mse < best_mse:  
            best_mse = mse
            best_model = model
    print(best_mse)
    return best_model

# appends results to end of file
def log_results(params, mse, r2, log_file="model_log.txt"):
    with open(log_file, 'a') as log_f:
        log_f.write(f"Hyperparameters: {params}\t MSE: {mse}, R-squared: {r2}\n")

# tests on other 20% of data
def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return r2, mse

def main():
    x_clean, y_encode = process_data()
    x_scaled = standardize(x_clean)
    x_train, x_test, y_train, y_test = split(x_scaled, y_encode)

    model = hyperparameter_tuning(x_train, y_train)
    r2, mse = evaluate(model, x_test, y_test)

    print(r2, mse)


if __name__ == "__main__":
    main()
