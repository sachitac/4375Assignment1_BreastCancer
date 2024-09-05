from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
  
def preprocess_data():
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
    print(breast_cancer_wisconsin_diagnostic.data.describe())

def main():
    preprocess_data()

if __name__=="__main__":
    main()
    
