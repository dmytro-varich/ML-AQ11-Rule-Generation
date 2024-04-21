# 💳 Predicting Term Deposit Subscriptions with AQ11 Algorithm

## Abstract
In this work in the field of machine learning, the AQ11 algorithm was implemented in Python, and its effectiveness was evaluated on a selected dataset for classification tasks. The main objective was to choose a suitable algorithm, prepare the data, implement it, and evaluate its effectiveness in the context of classifying clients based on their likelihood to subscribe to a term deposit, using provided data on their behavior in direct marketing campaigns. The experiments demonstrated that the implemented AQ11 algorithm achieves satisfactory performance in predicting clients' decisions to subscribe to a term deposit. The use of this algorithm provides hope for improving the process of evaluating potential clients in practice. These results have a significant impact on the financial services sector, helping to improve automated decision-making systems in banking and serving as a basis for further research in improving classification algorithms and their applications in the banking sector.

## Keywords
`aq11`, `machine-learning`, `supervised-learning`, `classification`, `python`, `business`, `banking`,
`marketing`, `term-deposit`.

## Used Libraries
1. **pandas**: A library for data analysis and manipulation. It was used for reading and processing bank client data.
2. **ucimlrepo**: A repository for accessing UCI Machine Learning datasets. It was used to fetch the dataset for the project.
3. **tabulate**: A library for formatting tabular data. It was used for tabulating the results.
4. **scikit-learn**: A machine learning library featuring various classification algorithms, as well as utilities for data preprocessing, model evaluation, and dataset splitting for training and testing. 
5. **collections**: A built-in Python library providing specialized container datatypes. It was used for defaultdict to handle missing values.

## AQ11 
AQ11 algorithm, introduced by Michalski in 1969, represents a fundamental principle of rule generation. The core idea revolves around generating decision rules for classifying data. Here's a summary of how the algorithm works:

**Initial Step**:
- The algorithm starts by selecting one positive example (E1) and one negative example (E2).
- The selected examples serve as initial prototypes for the positive and negative classes.

**Generation of Rules**:
- AQ11 uses a decision tree to iteratively expand the set of prototypes by generating new positive and negative examples.
- It employs a concept called 'counterexample' to identify relevant attributes and their values for refining the prototypes.
- The algorithm aims to maximize the coverage of positive examples while minimizing the coverage of negative examples.
- Additionally, it adheres to the absorption law to optimize the rule generation process.

**Rule Construction**:
- Rules are constructed based on the identified attributes and their values.
- The process involves selecting relevant attributes and their corresponding values to form rules that accurately represent the classification criteria.

**Refinement of Rules**:
- The generated rules are refined iteratively to improve their accuracy and coverage.
- This refinement process involves adjusting the rules based on the identified attributes and their values to better classify the examples.

**Finalization**:
- Once the rules are refined and optimized, they are used for classification tasks.
- The algorithm aims to generate rules that effectively classify new examples into predefined classes.

Overall, the AQ11 algorithm employs a systematic approach to rule generation, utilizing decision trees, iterative refinement, and the absorption law to produce accurate and reliable classification rules. For more detailed information about the algorithm, you can be found [here](https://kristina.machova.website.tuke.sk/pdf/SU4.pdf).

## Dataset
After selecting the AQ11 algorithm for implementation, the decision was made to find a suitable dataset that could clearly demonstrate the algorithm's operation and bring practical benefits. Initially, three data domains were considered: financial, social, and medical, but ultimately, the focus was directed towards the financial sector, considering previous experience with financial data in prior projects. The main data sources became platforms providing extensive databases, such as [Kaggle](https://www.kaggle.com/) and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). After analyzing the available datasets, the ["Bank Marketing"](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) dataset, associated with direct marketing campaigns of a Portuguese bank, was chosen. This CSV-formatted dataset contains a large amount of information: **45211** **instances**, **16 attributes**, and **1 target** attribute (Tab. 1). The classification goal is to predict whether a client (yes/no) will subscribe to a term deposit (target variable y). To implement the dataset into the custom code, the ucimlrepo library was utilized, allowing to bypass the need for dataset downloading.

| Name        | Role   | Type      | Description                                    |
|-------------|--------|-----------|------------------------------------------------|
| age         | Future | Integer   | Client's age.                                  |
| job         | Future | Categorical | Type of client's employment.                  |
| marital     | Future | Categorical | Client's marital status.                       |
| education   | Future | Categorical | Client's level of education.                   |
| default     | Future | Binary    | Client's credit default status.                |
| balance     | Future | Integer   | Client's account balance.                      |
| housing     | Future | Binary    | Information whether client has housing loan.   |
| loan        | Future | Binary    | Information whether client has personal loan.  |
| contact     | Future | Categorical | Method of communication with the client.      |
| day_of_week | Future | Date      | Last contact day of the week with the client.  |
| month       | Future | Date      | Last contact month with the client.            |
| duration    | Future | Integer   | Duration of the last contact with the client (in seconds). |
| campaign    | Future | Integer   | Number of contacts during this campaign for this client. |
| pdays    | Future | Integer    | Number of days passed since the client was last contacted in the previous campaign. |
| previous | Future | Integer    | Number of contacts performed before this campaign for this client. |
| poutcome | Future | Categorical | Outcome of the previous marketing campaign.              |
| y        | Target | Binary     | Target attribute indicating whether the client subscribed to a term deposit. |

## Implementing Algorithm
This section provides a concise overview of each individual function for a better understanding of the code itself, which is located in the file `aq11_algorithm.py`.

1. **Preprocess Data**: This function serves as a custom preprocessing tool, tailored to specific requirements rather than being universally applicable. It involves selecting attributes and potentially reducing the dataset size to prepare the DataFrame for further analysis.

2. **Separation Data**: This function takes a DataFrame, the number of rows, the target variable, and a list of attributes as parameters. Its primary purpose is to divide the data into training and testing datasets. The function first separates the data based on the target variable `'y'`, then samples the specified number of rows for each subset. Finally, it splits the data into ***training*** and ***testing*** sets using the `train_test_split` function. The function returns a dictionary containing the training and testing sets for both subsets `'E1'` and `'E2'`.

3. **AQ11 Algorithm**: This function serves as the core of the algorithm, where we either generate rules or obtain a dictionary of predicted data using the generated rules. Within the function, there are two auxiliary functions: `create_metadata` and `custom_absorption_law`. Creating metadata is essential for retaining attribute values, while the absorption function simplifies rules within a single example to all counterexamples. The principle of the algorithm is implemented in accordance with the materials provided above, which would require a substantial explanation of its logic. Notably, we utilized the `eval` method for checking attribute values against rules, despite its potential risks. Additionally, we employed the `join` method to concatenate conditions.

4. **Evaluation**: This function calculates various evaluation metrics based on the provided True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN) values. It computes `precision`, `recall`, `F1-score`, `accuracy`, and `error rate`. Additionally, it generates a confusion matrix table and a metrics table for visualization and interpretation of the evaluation results.

5. **Main**: The final function orchestrates the entire process, starting with fetching the dataset, performing data preprocessing, initializing parameters for data separation into training and testing sets, generating rules using the AQ11 algorithm, attempting to predict using the same method but with test data, and finally evaluating the performance.

## Conclusion 
The outcome of all this work can be considered satisfactory. In terms of code implementation, there may be some shortcomings, but overall, I did not find any glaring errors during the final checks, although it's possible that some issues could be attributed to human error on my part. However, the main concern lies with the algorithm itself, which is not the latest version of the AQ branch algorithm. Additionally, the AQ11 algorithm's ability to create overlapping boundaries between different classes in certain feature spaces may complicate result interpretation, potentially leading to confusion in determining an object's class membership. 

During the evaluation and hyperparameter tuning, such as the number of attributes, rows, and selection of the target variable, it can be noted that the algorithm achieves an **accuracy** between `50% to 70%` in **90%** of cases with `8 attributes` and `100 rows`. This is considered a reasonably good result, considering all the issues mentioned above.

Here is an example of the generated rules, confusion matrix, and performance metrics for `30 rows` with `8 attributes` that are outputted to the console to demonstrate the correctness of the algorithm's operation. 

```
Generated Rules:
(((job != 'student' or loan != 'no') and (age < 2 or marital != 'married' or loan != 'no') and (age < 1 or marital != 'married' or loan != 'no') and (duration < 2 or age < 2 or marital != 'married' or loan != 'no') and (marital != 'married' or age < 1 or loan != 'no' or duration < 2) and (age < 2 or marital != 'married' or balance < 2 or loan != 'no') and (marital != 'married' or age < 1 or loan != 'no' or balance < 2) and (education != 'tertiary' or age < 2 or marital != 'married' or loan != 'no') and (age < 1 or marital != 'married' or loan != 'no' or duration < 2 or job != 'technician')) or ((education != 'secondary') and (balance < 2 or education != 'secondary') and (age < 2 or job != 'retired' or marital != 'married' or duration < 2) and (job != 'retired' or age < 2 or marital != 'married' or education != 'secondary') and (education != 'tertiary' or job != 'retired' or age < 2 or marital != 'married') and (age < 1 or marital != 'married' or duration < 2 or education != 'secondary' or job != 'technician')) or ((balance < 2 or marital != 'married' or education != 'secondary') and (marital != 'married' or age > 0 or housing != 'no' or duration > 2) and (housing != 'no' or duration > 2 or age > 0 or marital != 'single') and (age < 2 or marital != 'married' or housing != 'no' or duration > 2 or education != 'secondary') and (job != 'retired' or marital != 'married' or age < 2 or housing != 'no' or duration > 2) and (job != 'retired' or age < 2 or housing != 'no' or duration > 2 or education != 'secondary') and (job != 'retired' or marital != 'married' or age < 2 or housing != 'no' or duration > 2 or education != 'secondary') and (job != 'retired' or marital != 'married' or age < 2 or duration > 2 or housing != 'no' or balance < 2) and (marital != 'single' or age > 0 or duration > 2 or housing != 'no' or job != 'student' or education != 'secondary')))

Confusion Matrix: 
+---------------+----+----------------+---+
| True Positive | 14 | False Negative | 1 |
| True Negative | 6  | False Positive | 9 |
+---------------+----+----------------+---+

Performance Metrics: 
+------------+-----+
| Precision  | 61% |
|   Recall   | 93% |
|  F1-Score  | 74% |
|  Accuracy  | 67% |
| Error Rate | 33% |
+------------+-----+
```

## Author 
My name is Dmytro Varich, and I am a student at [TUKE](https://www.tuke.sk/wps/portal) University, majoring in Intelligent Systems. This documentation is intended for the completion of Assignment 1 in the subject of Machine Learning. Similar content is also shared on my [Telegram](https://t.me/varich_channel) channel.

Email: dmytro.varich@student.tuke.sk

This documentation was also written with the intention of delving deeper into the field of Machine Learning.

