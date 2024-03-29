# import libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

def preprocess_data(dataset):
    #
    df = dataset.data.original.copy()

    #
    selected_categories = ['technician', 'entrepreneur', 'student', 'self-employed', 'retired']
    df = df.loc[df['job'].isin(selected_categories)]
    
    #
    bins = [17, 44, 59, 74, 100]
    labels = [0, 1, 2, 3]
    df['age'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    #
    bins = [float('-inf'), 0, 5000, 10000, 15000, 20000, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    df['balance'] = pd.cut(df['balance'], bins=bins, labels=labels)

    #
    bins = [0, 60, 300, 900, 1800, float('inf')]
    labels = [0, 1, 2, 3, 4]
    df['duration'] = pd.cut(df['duration'], bins=bins, labels=labels)

    #
    categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    for column in categorical_attributes:
        df[column] = df[column].astype('category').cat.codes

    #
    df = df.dropna()

    return df

def separation_data(data) -> dict:
    #
    my_attributes = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'contact', 'duration']
    
    #
    E1 = data[data['y'] == 1][my_attributes]
    E2 = data[data['y'] == 0][my_attributes]

    # 
    E1 = E1.sample(n=40)
    E2 = E2.sample(n=40)

    # 
    E1_train, E1_test = train_test_split(E1, test_size=0.5)
    E2_train, E2_test = train_test_split(E2, test_size=0.5)

    return {'E1': (E1_test, E1_train), 'E2': (E2_test, E2_train)}

def aq11(E1, E2, Generated_Rules=None):
    if Generated_Rules is None:
        E1_E2_rules = str()
        for E1_index, E1_row in E1.iterrows():
            E1_rules = list()
            # condition 
            # if ... then continue else ... mb down cykle
            for E2_index, E2_row in E2.iterrows():
                generate_rules_for_row: list[str] = list()
                for E1_column, E1_value, _, E2_value in zip(E1_row.items(), E2_row.items()):
                    if E1_value > E2_value:
                        generate_rules_for_row.append(f"{E1_column} > {E2_value}")                
                    elif E1_value < E2_value: 
                        generate_rules_for_row.append(f"{E1_column} < {E2_value}") 
                    else:
                        continue
                E1_rules.append(generate_rules_for_row)

            # for rules in E1_rules:
            #     for i=1, rule in rules:
                        
            # E1_E2_rules.append()
                           
    else: 
        pass # train data -> evaluation the models

def main():
    # 
    bank_marketing = fetch_ucirepo(id=222) 

    # 
    processed_dataset = preprocess_data(bank_marketing) 
    separated_data = separation_data(processed_dataset)

    # 
    E1_test, E2_test = separated_data['E1'][0], separated_data['E2'][0]
    Generated_Rules = aq11(E1_test, E2_test)

    # 
    # E1_train, E2_train = separated_data['E1'][1], separated_data['E2'][1]
    # aq11(E1_train, E2_train, Generated_Rules)


if __name__ == '__main__':
    main()