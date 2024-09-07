from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


def process_data():
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    x = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # remove null values
    x_clean = x.dropna()  # remove missing values from row x
    y_clean = y.loc[x_clean.index]  # align row y with row x
    x_clean = x_clean.drop_duplicates()  # remove duplicates
    y_clean = y_clean.loc[x_clean.index]  # align row y with clean x

    # convert categorical values to numerical
    label_encoder = LabelEncoder()
    y_encode = label_encoder.fit_transform(y_clean.values.ravel())

    return x_clean, y_encode  # return cleaned data for further processing

def standardize(x_clean):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)  # scale the features
    return x_scaled

def split(x_clean, y_encode):
    x_train, x_test, y_train, y_test = train_test_split(x_clean, y_encode, random_state=104,  test_size=0.25,  shuffle=True) 
    return x_train, x_test, y_train, y_test

def train(x_train, y_train):
    sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate="invscaling")
    sgd_regressor.fit(x_train, y_train)
    return sgd_regressor

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def main():
    x_clean, y_encode = process_data()
    x_scaled = standardize(x_clean) 
    x_train, x_test, y_train, y_test = split(x_scaled, y_encode) 
    model = train(x_train, y_train)
    r2 = evaluate(model, x_test, y_test)
    print(r2)

if __name__ == "__main__":
    main()

