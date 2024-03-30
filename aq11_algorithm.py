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

    return {'E1': (E1_train, E1_test), 'E2': (E2_train, E2_test)}

def aq11(E1, E2, Generated_Rules=None):
    def create_metadata(row) -> None:
        metadata = dict()
        for column, value in row.items():
            metadata[column] = value 
        return metadata

    if not Generated_Rules:
        rules_between_E1_and_E2 = list()
        for E1_index, E1_row in E1.iterrows():
            E1_generated_rules = list()
            is_covered = False
            for _, E2_row in E2.iterrows():
                generate_rules_for_row = str()

                for (E1_column, E1_value), (_, E2_value) in zip(E1_row.items(), E2_row.items()):
                    
                    if E1_index > 0:
                        metadata = create_metadata(E1_row)
                        
                        if eval(' or '.join(rules_between_E1_and_E2), None, metadata): 
                            is_covered = True
                            break 
                     
                    if E1_value > E2_value:
                        generate_rules_for_row += f"{E1_column} > {E2_value} or "                
                    elif E1_value < E2_value: 
                        generate_rules_for_row += f"{E1_column} < {E2_value} or " 
                    else:
                        continue

                E1_generated_rules.append("(" + generate_rules_for_row[:-4] + ")")
            if not is_covered:
                rules_between_E1_and_E2.append("(" + ' and '.join(E1_generated_rules) + ")")

        Generated_Rules = ' or '.join(rules_between_E1_and_E2)
        return Generated_Rules
    else: 
        # 
        metrics_dict = {
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
                metrics_dict["TN"] += 1
        
        #
        for _, E2_row in E2.iterrows():
            metadata = create_metadata(E2_row)

            if eval(Generated_Rules, None, metadata):
                metrics_dict["FP"] += 1
            else: 
                metrics_dict["FN"] += 1
        
        # 
        return metrics_dict

def evaluate_metrics(TP, TN, FP, FN) -> None:
    total_metrics_sum = sum((TP, TN, FP, FN))

    #
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / total_metrics_sum
    error_rate = (FP + FN) / total_metrics_sum

    # 
    print(f"Precision: {precision * 100:.3f}%")
    print(f"Recall: {recall * 100:.3f}%")
    print(f"F1-Score: {f1_score * 100:.3f}%")
    print(f"Accuracy: {accuracy * 100:.3f}%")
    print(f"Error Rate: {error_rate * 100:.3f}%")

def main():
    # 
    bank_marketing = fetch_ucirepo(id=222) 

    # 
    processed_dataset = preprocess_data(bank_marketing) 
    separated_data = separation_data(processed_dataset)

    # 
    E1_train, E2_train = separated_data['E1'][0].reset_index(drop=True), separated_data['E2'][0].reset_index(drop=True)
    Generated_Rules = aq11(E1_train, E2_train)

    # 
    E1_test, E2_test = separated_data['E1'][1].reset_index(drop=True), separated_data['E2'][1].reset_index(drop=True)
    metrics_data = aq11(E1_test, E2_test, Generated_Rules)

    # 
    evaluate_metrics(metrics_data['TP'], metrics_data['TN'], metrics_data['FP'], metrics_data['FN'])  


if __name__ == '__main__':
    main()