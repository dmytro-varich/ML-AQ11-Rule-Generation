# import libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from tabulate import tabulate 
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

def separation_data(data, num_row) -> dict:
    #
    my_attributes = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'contact', 'duration']
    
    #
    E1 = data[data['y'] == 1][my_attributes]
    E2 = data[data['y'] == 0][my_attributes]

    # 
    E1 = E1.sample(n=num_row)
    E2 = E2.sample(n=num_row)

    # 
    E1_train, E1_test = train_test_split(E1, test_size=0.5)
    E2_train, E2_test = train_test_split(E2, test_size=0.5)

    return {'E1': (E1_train, E1_test), 'E2': (E2_train, E2_test)}

def aq11(E1, E2, Generated_Rules: str = None) -> str | dict[str: int]:
    #
    def create_metadata(row) -> None:
        metadata = dict()
        for column, value in row.items():
            metadata[column] = value 
        return metadata
        
    # 
    if not Generated_Rules:
        #
        rules_between_E1_and_E2 = set()
        
        #
        for _, E1_row in E1.iterrows():
            # 
            E1_generated_rules = set()

            # 
            metadata = create_metadata(E1_row)
            if eval("(" + ' or '.join(rules_between_E1_and_E2) + ")", None, metadata):
                continue

            # 
            for _, E2_row in E2.iterrows():
                generate_rules_for_row = str()

                # 
                for E2_column, E2_value in E2_row.items():
                    
                    if E1_row[E2_column] > E2_value:
                        generate_rules_for_row += f"{E2_column} > {E2_value} or "                
                    elif E1_row[E2_column] < E2_value: 
                        generate_rules_for_row += f"{E2_column} < {E2_value} or " 
                
                if generate_rules_for_row:
                    E1_generated_rules.add("(" + generate_rules_for_row[:-4] + ")")
      
            rules_between_E1_and_E2.add("(" + ' and '.join(E1_generated_rules) + ")")

        Generated_Rules: str = "(" + ' or '.join(rules_between_E1_and_E2) + ")"
        return Generated_Rules
    else: 
        # 
        metrics_dict: dict = {
            "TP": 0, 
            "TN": 0, 
            "FP": 0, 
            "FN": 0 
        }

        #
        for _, E1_row in E1.iterrows():
            metadata = create_metadata(E1_row)

            if eval(Generated_Rules, None, metadata):
                metrics_dict["TP"] += 1
            else: 
                metrics_dict["FN"] += 1
        
        #
        for _, E2_row in E2.iterrows():
            metadata = create_metadata(E2_row)

            if eval(Generated_Rules, None, metadata):
                metrics_dict["FP"] += 1
            else: 
                metrics_dict["TN"] += 1
        
        # 
        return metrics_dict

def evaluate_metrics(TP: int, TN: int, FP: int, FN: int) -> None:
    #
    total_metrics_sum = sum((TP, TN, FP, FN))
    
    #
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / total_metrics_sum
    error_rate = (FP + FN) / total_metrics_sum

    #
    metrics_data = [
        ("True Positive", TP, "False Negative", FN), 
        ("True Negative", TN, "False Positive", FP)
    ]
    confusion_matrix_table = tabulate(metrics_data, tablefmt="pretty")


    metrics_data = [
        ("Precision",  f"{precision * 100:.0f}%"),
        ("Recall",     f"{recall * 100:.0f}%"),
        ("F1-Score",   f"{f1_score * 100:.0f}%"), 
        ("Accuracy",   f"{accuracy * 100:.0f}%"),
        ("Error Rate", f"{error_rate * 100:.0f}%")
    ]    
    metrics_table = tabulate(metrics_data, tablefmt="pretty")


    #
    print(f"Confusion Matrix: \n{confusion_matrix_table}\n")
    print(f"Performance Metrics: \n{metrics_table}")

def main():
    # 
    bank_marketing = fetch_ucirepo(id=222) 
    
    # 
    processed_dataset = preprocess_data(bank_marketing) 
    separated_data = separation_data(processed_dataset, 40)

    # 
    E1_train, E2_train = separated_data['E1'][0].reset_index(drop=True), separated_data['E2'][0].reset_index(drop=True)
    Generated_Rules = aq11(E1_train, E2_train)
    print("Generated Rules:\n" + Generated_Rules + "\n")

    # 
    E1_test, E2_test = separated_data['E1'][1].reset_index(drop=True), separated_data['E2'][1].reset_index(drop=True)
    metrics_data = aq11(E1_test, E2_test, Generated_Rules)

    # 
    evaluate_metrics(metrics_data['TP'], metrics_data['TN'], metrics_data['FP'], metrics_data['FN'])  

if __name__ == '__main__':
    main()