from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

    # display first 5 rows of clean data
    print("First 5 rows of x_cleaned\n", x_clean.head())
    print("First 5 rows of y encoded\n", y_encode[:5])

    return x_clean, y_encode  # return cleaned data for further processing

def standardize(x_clean):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)  # scale the features
    print("First 5 rows of standardized x\n", x_scaled[:5])
    return x_scaled

def main():
    x_clean, y_encode = process_data()  # get cleaned data
    x_scaled = standardize(x_clean)  # standardize features

if __name__ == "__main__":
    main()

