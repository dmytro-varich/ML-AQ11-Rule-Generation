# Import libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from tabulate import tabulate 
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Function to preprocess the data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the DataFrame
    df = df.copy()

    # Select specific categories in the 'job' column
    selected_categories = ['technician', 'entrepreneur', 'student', 'retired']
    df = df.loc[df['job'].isin(selected_categories)]
    
    # Convert 'age' into categorical bins
    bins = [17, 44, 59, 74, float('inf')]
    labels = [0, 1, 2, 3]
    df['age'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # Convert 'balance' into categorical bins
    bins = [float('-inf'), 0, 5000, 20000, float('inf')]
    labels = [0, 1, 2, 3]
    df['balance'] = pd.cut(df['balance'], bins=bins, labels=labels)

    # Convert 'duration' into categorical bins
    bins = [0, 60, 300, 900, float('inf')]
    labels = [0, 1, 2, 3]
    df['duration'] = pd.cut(df['duration'], bins=bins, labels=labels)

    # Convert target attribute into numeric codes
    df['y'] = df['y'].astype('category').cat.codes

    # Drop any rows with missing values
    df = df.dropna()
    return df

# Function to separate data into two sets
def separation_data(data: pd.DataFrame, num_row: int, target: int, attributes: list[str]) -> dict[str: pd.DataFrame]:
    # Separate data based on target variable 'y'
    E1 = data[data['y'] == target][attributes]
    E2 = data[data['y'] == int(not target)][attributes]

    # Sample the data to specified number of rows
    E1 = E1.sample(n=num_row)
    E2 = E2.sample(n=num_row)

    # Split data into train and test sets
    E1_train, E1_test = train_test_split(E1, test_size=0.5)
    E2_train, E2_test = train_test_split(E2, test_size=0.5)

    return {'E1': (E1_train, E1_test), 'E2': (E2_train, E2_test)}

# Function to generate rules or evaluate metrics based on given data
def aq11(E1: pd.DataFrame, E2: pd.DataFrame, Generated_Rules: str = None) -> str | dict[str: int]:
    # Function to create metadata from a DataFrame row
    def create_metadata(row: pd.Series) -> dict:
        metadata = dict()
        for column, value in row.items():
            metadata[column] = value 
        return metadata
    
    # Absorption law function to simplify rules
    def custom_absorption_law(rules: list[tuple[str]]) -> str:
        # Initialize an empty list to store simplified rules
        simplified_rules = []

        # Iterate through each rule in the list
        for i, rule_i in enumerate(rules):
            # Convert the rule to a set for efficient comparison
            rule_set = set(rule_i)

            # Initialize a defaultdict to store updated conditions
            update_dict_condition = defaultdict(str)

            # Iterate through the remaining rules in the list
            for rule_j in rules[i+1:]:
                # Count the number of common conditions between two rules
                common_condition_count = 0
                s_rule = set(rule_j)
                for condition1 in rule_set:
                    if condition1 in s_rule:
                        common_condition_count += 1
                    else: 
                        attribute1, sign1, value1 = condition1.split()
                        for condition2 in s_rule:
                            attribute2, sign2, value2 = condition2.split()
                            # Absorption of conditions if they have the same attribute and sign, but different values
                            if attribute1 == attribute2 and sign1 == sign2 == ">" and value1 != value2:
                                max_val = max(int(value1), int(value2))
                                common_condition_count += 1
                                update_dict_condition[condition1] = f"{attribute1} {sign1} {max_val}"
                            # Absorption of conditions if they have the same attribute and sign, but different values
                            elif attribute1 == attribute2 and sign1 == sign2 == "<" and value1 != value2:
                                min_val = min(int(value1), int(value2))
                                common_condition_count += 1
                                update_dict_condition[condition1] = f"{attribute1} {sign1} {min_val}"

                # If all conditions in the rule set are common, absorb the rule  
                if len(rule_set) == common_condition_count:
                    temp_rule_set = rule_set.copy()
                    for key, value in update_dict_condition.items():
                        temp_rule_set.remove(key)
                        temp_rule_set.add(value)
                    # Add the updated rule set to the list if it's not already present 
                    if temp_rule_set not in simplified_rules:
                        simplified_rules.append(temp_rule_set)
                        
        # If no rules are simplified, return the original rules
        if not simplified_rules:
            simplified_rules = rules.copy()

        # Convert the result to a string
        result_str = ""
        for i, rule in enumerate(simplified_rules):
            rule_str = " or ".join(rule)
            if i == 0:
                result_str += f"({rule_str})"
            else:
                result_str += f" and ({rule_str})"
        
        return result_str 
        
    # Generate rules if none provided, else evaluate metrics
    if not Generated_Rules:
        # Initialize set to store generated rules
        rules_between_E1_and_E2 = set()
        
        # Iterate through each row in E1
        for _, E1_row in E1.iterrows():
            generate_rules_for_row: set[tuple[str]] = set()
            
            '''
            Create metadata for current E1 row.
            If any of the rules are satisfied by the metadata of the current row,
            then continue to the next iteration without generating new rules.
            '''
            metadata = create_metadata(E1_row)
            if eval("(" + ' or '.join(rules_between_E1_and_E2) + ")", None, metadata):
                continue

            # Iterate through each row in E2
            for _, E2_row in E2.iterrows():
                # Initialize an empty set to store generated rules for the current row
                set_rules_for_value = list()
                
                # Iterate through each column and value in the current E2 row
                for E2_column, E2_value in E2_row.items():
                    # Check if the attribute type is object
                    if E2[E2_column].dtype == object:
                        # If so, compare values as strings 
                        if E1_row[E2_column] != E2_value:
                            set_rules_for_value.append(f"{E2_column} != '{E2_value}'")
                    else: 
                        # Generate a rule based on the comparison between E1 and E2 values
                        if E1_row[E2_column] > E2_value:
                            set_rules_for_value.append(f"{E2_column} > {E2_value}")                
                        elif E1_row[E2_column] < E2_value: 
                            set_rules_for_value.append(f"{E2_column} < {E2_value}") 
                
                # Add the generated rules for the current row to the set of rules
                if set_rules_for_value: 
                    generate_rules_for_row.add(tuple(set_rules_for_value))
            
            # Apply custom absorption law to simplify the generated rules
            rules_after_absorb = custom_absorption_law(sorted(generate_rules_for_row, key=len))

            # Add the simplified rules to the set of rules between E1 and E2
            rules_between_E1_and_E2.add("(" + rules_after_absorb + ")")
            
        # Combine all generated rules into a single string
        Generated_Rules: str = "(" + ' or '.join(rules_between_E1_and_E2) + ")"

        # Return the generated rules
        return Generated_Rules
    else: 
        # Initialize a dictionary to store the metrics
        metrics_dict: dict = {
            "TP": 0,    # True Positive
            "TN": 0,    # True Negative
            "FP": 0,    # False Positive
            "FN": 0     # False Negative
        }

        # Iterate through each row in E1 DataFrame to calculate True Positives (TP) and False Negatives (FN)
        for _, E1_row in E1.iterrows():
            # Create metadata for the current row in E1 DataFrame.
            metadata = create_metadata(E1_row)

            # Evaluate the generated rules using the metadata
            if eval(Generated_Rules, None, metadata):
                metrics_dict["TP"] += 1
            else: 
                metrics_dict["FN"] += 1
        
        # Iterate through each row in E2 DataFrame to calculate False Positives (FP) and True Negatives (TN)
        for _, E2_row in E2.iterrows():
            # Create metadata for the current row in E2 DataFrame
            metadata = create_metadata(E2_row)

            # Evaluate the generated rules using the metadata
            if eval(Generated_Rules, None, metadata):
                metrics_dict["FP"] += 1
            else: 
                metrics_dict["TN"] += 1
        
        # Return the metrics dictionary containing TP, TN, FP, FN
        return metrics_dict

# Function to calculate and display evaluation metrics
def evaluate_metrics(TP: int, TN: int, FP: int, FN: int) -> None:
    # Calculate metrics
    total_metrics_sum = sum((TP, TN, FP, FN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / total_metrics_sum
    error_rate = (FP + FN) / total_metrics_sum

    # Create confusion matrix table
    metrics_data = [
        ("True Positive", TP, "False Negative", FN), 
        ("True Negative", TN, "False Positive", FP)
    ]
    confusion_matrix_table = tabulate(metrics_data, tablefmt="pretty")

    # Create metrics table
    metrics_data = [
        ("Precision",  f"{precision * 100:.0f}%"),
        ("Recall",     f"{recall * 100:.0f}%"),
        ("F1-Score",   f"{f1_score * 100:.0f}%"), 
        ("Accuracy",   f"{accuracy * 100:.0f}%"),
        ("Error Rate", f"{error_rate * 100:.0f}%")
    ]    
    metrics_table = tabulate(metrics_data, tablefmt="pretty")

    # Display confusion matrix and metrics
    print(f"Confusion Matrix: \n{confusion_matrix_table}\n")
    print(f"Performance Metrics: \n{metrics_table}")

# Main function to orchestrate the entire process
def main() -> None:
    # Fetch dataset
    bank_marketing = fetch_ucirepo(id=222) 
    dataset = bank_marketing.data.original

    # Define the target variable
    target = 0 

    # Attributes to consider
    my_attributes = ['age', 'job', 'marital', 'education', 'balance', 'loan', 'housing', 'duration']

    # Preprocess the dataset and separate data into training and testing sets
    processed_dataset = preprocess_data(dataset)
    separated_data = separation_data(processed_dataset, 100, target, my_attributes)

    # Train and generate rules
    E1_train, E2_train = separated_data['E1'][0].reset_index(drop=True), separated_data['E2'][0].reset_index(drop=True)
    Generated_Rules = aq11(E1_train, E2_train)

    # Display generated rules
    print("Generated Rules:\n" + Generated_Rules + "\n")

    # Test and evaluate metrics
    E1_test, E2_test = separated_data['E1'][1].reset_index(drop=True), separated_data['E2'][1].reset_index(drop=True)
    metrics_data = aq11(E1_test, E2_test, Generated_Rules)

    # Display evaluation metrics. 
    evaluate_metrics(metrics_data['TP'], metrics_data['TN'], metrics_data['FP'], metrics_data['FN'])  

# Execute main function
if __name__ == '__main__':
    main()