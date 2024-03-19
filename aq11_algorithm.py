# import libraries
from ucimlrepo import fetch_ucirepo 

def main():
    # fetch dataset 
    bank_marketing = fetch_ucirepo(id=222) 
    
    # data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 
    
    # metadata 
    print(bank_marketing.metadata) 
    
    # variable information 
    print(bank_marketing.variables) 

    print(X)
    print(y)

if __name__ == '__main__':
    main()
