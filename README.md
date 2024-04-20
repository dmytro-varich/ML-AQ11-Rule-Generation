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
After selecting the AQ11 algorithm for implementation, the decision was made to find a suitable dataset that could clearly demonstrate the algorithm's operation and bring practical benefits. Initially, three data domains were considered: financial, social, and medical, but ultimately, the focus was directed towards the financial sector, considering previous experience with financial data in prior projects. The main data sources became platforms providing extensive databases, such as [Kaggle](https://www.kaggle.com/) and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). After analyzing the available datasets, the ["Bank Marketing"](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) dataset, associated with direct marketing campaigns of a Portuguese bank, was chosen. This CSV-formatted dataset contains a large amount of information: 45211 instances, 16 attributes, and 1 target attribute (Tab. 1). The classification goal is to predict whether a client (yes/no) will subscribe to a term deposit (target variable y). To implement the dataset into the custom code, the ucimlrepo library was utilized, allowing to bypass the need for dataset downloading.

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
...

## Conclusion 
... 

## Author 
My name is Dmytro Varich, and I am a student at [TUKE](https://www.tuke.sk/wps/portal) University, majoring in Intelligent Systems. This documentation is intended for the completion of Assignment 1 in the subject of Machine Learning. Similar content is also shared on my [Telegram](https://t.me/varich_channel) channel.

Email: dmytro.varich@student.tuke.sk

This documentation was also written with the intention of delving deeper into the field of Machine Learning.

