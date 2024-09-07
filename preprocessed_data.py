from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def process_data():
# fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    x = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    #remove null values
    x_clean = x.dropna() #remove missing values from row x
    y_clean = y.loc[x_clean.index] #align row y with row x
    x_clean = x_clean.drop_duplicates()
    y_clean = y_clean.loc[x_clean.index]

    #convert categorical values to numerical
    label_encoder = LabelEncoder()
    y_encode = label_encoder.fit_transform(y_clean.values.ravel())

    #display first 5 rows of clean data
    print("print 5 rows of x_cleaned\n", x_clean.head())
    print("first 5 rows of y encoded\n", y_encode[:5])

def main():
    process_data()

if __name__=="__main__":
    main()
