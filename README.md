## Binary classification for customer response
Semester 2, 2017 

- data: dataset from a clothing store chain that uses different marketing strategies to promote sales
- process: 
- 1) Cleaning data
- 2) EDA and Feature engineering
- 3) Modeling (Logistic, SVM, XGBoost, LDA and QDA)
- 4) Model evaluation
- 5) Further discussion
          
- results: [report](https://github.com/YiranJing/CrossSectionalAnalysis/blob/master/ClassificationAnalyis/CustomerResponseClassification/Report.pdf)

The effect of profit per customer on finnacial statement:
<img width="770" alt="Screen Shot 2019-06-29 at 9 42 03 pm" src="https://user-images.githubusercontent.com/31234892/60383639-e0567a80-9ab6-11e9-8b9b-31f520552418.png">


#### DataSet description
This assignment is based on the dataset from a clothing store chain that uses different marketing strategies to promote sales
#### business objective 
Modelling Consumer Response to Marketing. classify which customers will respond to direct mail marketing based on data collected for past customers. **Business success criteria**: based on cost-benefit table we built.

###  Data understanding (EDA)
- Correlation matrix
- Important features relative to target
- Distribution of Target. (unbalanced)
- Distribution of categorical variables.
- Distribution of continuous variables.

### Data Cleaning and feature Engineering
- Cleaning data
 - check duplicates and missing values (**Verify data quality**)
 - omitted customer_id from further analysis
 - convert categorical variables to numerical variables.
- Data Transformation
 - used to stabilize variance, make the continuous variables more normal-like

### Modelling
- Standardise Predictors
- Define Threshold based on the cost-benefit table (**business success criteria**)
- Modeling
 - Logistic regression, with ridge, lasso
 - Support vector machines
 - XGBoost
 - Gaussian discriminant analysis (LDA and QDA)

### Evaluation
 - benchmark
 - calculate F1, recall and recision
 - ROC plot
 - Precision recall curve
 - **Bootstrap** for comfidence interval
 - Significant Coefficient Histogram for L1

## Author
- Saad Ahmed
- Maha Siraj
