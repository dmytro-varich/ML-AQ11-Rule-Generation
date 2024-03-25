# import libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

def preprocess_data(dataset):
    df = dataset.data.original

    bins = [17, 44, 59, 74, 100]
    labels = [0, 1, 2, 3]
    df['age'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    bins = [-8019, 0, 5000, 10000, 15000, 20000, 102127]
    labels = [0, 1, 2, 3, 4, 5]
    df['balance'] = pd.cut(df['balance'], bins=bins, labels=labels)

    bins = [0, 60, 120, 300, 600, 900, 1800, 3600, 4918]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    df['duration'] = pd.cut(df['duration'], bins=bins, labels=labels)

    categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    for column in categorical_attributes:
        df[column] = df[column].astype('category').cat.codes

    df = df.dropna()

    return df

def separation_data(data) -> dict:
    my_attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration']
    E1 = data[data['y'] == 1][my_attributes][:80]
    E2 = data[data['y'] == 0][my_attributes][:80]
    print(E1)
    print(E2)
    E1_train, E1_test = train_test_split(E1, test_size=0.5, random_state=42)
    E2_train, E2_test = train_test_split(E2, test_size=0.5, random_state=42)

    return {'E1': (E1_train, E1_test), 'E2': (E2_train, E2_test)}

def aq11():
    pass

def main():
    # 
    bank_marketing = fetch_ucirepo(id=222) 

    # 
    processed_dataset = preprocess_data(bank_marketing) 
    print(processed_dataset)
    # 
    separated_data = separation_data(processed_dataset)

if __name__ == '__main__':
    main()