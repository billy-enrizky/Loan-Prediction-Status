# Loan Prediction Project

## 1. Dataset Information

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('loan_prediction.csv')

# Display value counts for each column
for col in df.columns:
    counts = df[col].value_counts()
    print(f'df[{col}]')
    print(counts)
    print('\n')
```

From the result above, we can summarize the dataset description:

- `Loan_ID`: Unique Loan ID
- `Gender`: Male/Female
- `Married`: Applicant married (Y/N)
- `Dependents`: Number of dependents
- `Education`: Applicant Education (Graduate/Under Graduate)
- `Self_Employed`: Self-employed (Y/N)
- `ApplicantIncome`: Applicant income
- `CoapplicantIncome`: Coapplicant income
- `LoanAmount`: Loan amount in thousands of dollars
- `Loan_Amount_Term`: Term of loan in months
- `Credit_History`: Credit history meets guidelines (Yes/No)
- `Property_Area`: Urban/Semi-Urban/Rural
- `Loan_Status`: Loan approved (Y/N) (This is the target variable)

## 2. Handling Missing Values

Percentage of Missing Values in each column:

```python
df.isnull().sum() * 100 / len(df)
```

Filling the missing values in columns with the mode:

```python
# Exclude Loan_ID for Model Building
df = df.drop('Loan_ID', axis=1)

# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
```

## 3. Data Exploratory

We aim to delve into and analyze the visual representation of data through plots, utilizing the seaborn library to create various graphical representations.

### Loan Amount Histogram

```python
(df['LoanAmount']).hist(bins=20);
```

The distribution of loan amounts in the dataset can be summarized as follows: The most common loan amount is $120.0, occurring 20 times in the data. This is followed by $110.0, which appears 17 times, and $100.0, occurring 15 times. The distribution shows a range of loan values, with some values occurring only once, such as $240.0, $214.0, $59.0, $166.0, and $253.0. Overall, there are 203 unique loan amounts in the dataset.

### Applicant Income Histogram

```python
print(df['ApplicantIncome'].value_counts())
(df['ApplicantIncome']).hist(bins=20);
```

The distribution of Applicant Income in the column `df[ApplicantIncome]` varies across a range of values. The majority of applicants have an income around $2,500, with 9 instances, followed closely by $4,583 and $6,000, each occurring 6 times. These values represent the most common income levels among applicants. As the income values increase, the frequency decreases, indicating that fewer applicants have higher incomes. This distribution provides insights into the income levels of applicants seeking loans.

### Coapplicant Income

```python
print(df['CoapplicantIncome'].value_counts())
(df['CoapplicantIncome']).hist(bins=20);
```

The distribution of coapplicant income in the dataset varies across a range of values. The most common coapplicant income is $0.0 (having no income), occurring 273 times in the dataset. There are several other values, such as $2500.0, $2083.0,$1666.0, and so on, each appearing 5 times. These observations suggest that a significant portion of coapplicants have no income, while other values are relatively evenly distributed. Overall, there are 287 unique values in the "CoapplicantIncome" column.

### Countplot of Gender

```python
import seaborn as sns
print(df['Gender'].value_counts())
sns.countplot(data=df, x='Gender');
```

The distribution of gender, as seen in the DataFrame (df), is summarized as follows: Among the applicants, 489 are male, while 112 are female, with the "Gender" column serving as the identifier for these categories.

### Countplot of Married People

```python
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df);
```

The distribution of marital status in the dataset can be summarized as follows: there are 398 individuals marked as "Married" and 213 individuals marked as "Not Married" (i.e., "No") in the "Married" column of the DataFrame.

### Countplot of Dependents

```python
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df);
```

The distribution of Dependents varies across different levels of dependents in the dataset. The majority of applicants have no dependents, with a total of 345 instances falling into this category. For those with one dependent, there are 102 cases, while households with two dependents have 101 instances. Finally, applicants with three or more dependents constitute the smallest group, with a count of 51. This breakdown provides valuable insights into how coapplicant income is distributed among applicants based on their dependent status.

### Countplot of Property Area

```python
print(df['Property_Area'].value_counts())
sns.countplot(x='Property_Area', data=df);
```

The distribution of property areas in the DataFrame (df) can be summarized as follows: There are 233 properties located in Semiurban areas, 202 in Urban areas, and 179 in Rural areas. This breakdown provides insights into the distribution of properties across different types of areas, which can be valuable for various analytical purposes.

## 4. Label Encoding

### Mapping to Numerical Values

```python
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype('int')
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0}).astype('int')
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
df['Property_Area'] = df['Property_Area'].map({'Rural':0,'Semi

urban':1,'Urban':2}).astype('int')
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0}).astype('int')
```

### Changing Categorical Entries to Numerical Entries

```python
df['Dependents'].value_counts()
# Replace '3+' with '4' to convert to integer
df['Dependents'] = df['Dependents'].replace(to_replace='3+', value='4')
```

## 8. Save the feature columns as vector X and the target column as vector y.

This would simplify the process of using `train_test_split` to construct the machine learning model.

```python
X, y = df.drop('Loan_Status', axis=1), df['Loan_Status']
```

## 9. Feature Scaling

### Standardizing Entries that have big values

```python
X.sample(5)
```

Kindly keep in mind that the values in the columns `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, and `Loan_Amount_Term` display notable deviations from the rest of the columns. To mitigate any potential bias within the Machine Learning Model, it is crucial to standardize these values. This standardization process can be effectively accomplished by employing the `StandardScaler` object from the `sklearn` module.

```python
from sklearn.preprocessing import StandardScaler

cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
X[cols] = StandardScaler().fit_transform(X[cols])
```

## 10. Dividing the dataset into a training set and a test set and then applying K-Fold Cross Validation.

We are in the process of creating a universal function that can be used with all machine learning models.

```python
from sklearn.model_selection import train_test_split, cross_val_score

model_df = {}

def model_evaluation(model, train_data = X, test_data = y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model.fit(X_train, y_train)
    print(f"The accuracy of the {model} model is {model.score(X_test, y_test)}")
    print(f"The average cross-validation score for the {model} model is {np.mean(cross_val_score(model, X, y))}")
    model_df[model] = round(np.mean(cross_val_score(model, X, y)) * 100, 2)
```

## 11. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model_evaluation(LogisticRegression(), X, y)
```

The accuracy of the `LogisticRegression()` model is 0.8537 (rounded to 4 decimal places).

## 12. Support Vector Classifier

```python
from sklearn import svm

model_evaluation(svm.SVC(kernel='rbf'), X, y)
```

The accuracy of the `SVC()` model is 0.7805 (rounded to 4 decimal places).

## 13. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

model_evaluation(DecisionTreeClassifier(), X, y)
```

The accuracy of the `DecisionTreeClassifier()` model is 0.6829 (rounded to 4 decimal places).

## 14. Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = GradientBoostingClassifier().fit(X_train, y_train)
model_evaluation(GradientBoostingClassifier(), X, y)
```

The accuracy of the `GradientBoostingClassifier()` model is 0.7724 (rounded to 4 decimal places).

### Analyzing The Features Importance

```python
feature_imp = dict(zip(clf.feature_names_in_, clf.feature_importances_))
feature_imp = {k: v for k,v in sorted(feature_imp.items(), key = lambda x:x[1], reverse=True)}
feature_imp
```

In summary, the most critical factors affecting loan approval are the applicant's credit history, income (both applicant and coapplicant), and the requested loan amount. These variables are considerably more influential than other factors such as loan term, property area, education, marital status, number of dependents, self-employment status, and gender. This information can guide decision-makers and help prioritize which features to focus on when assessing loan applications.

## 15. Random Tree Classifier

```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier().fit(X_train, y_train)
model_evaluation(RandomForestClassifier(), X, y)
```

The accuracy of the `RandomForestClassifier()` model is 0.7967 (rounded to 4 decimal places).

### Analyzing The Feature Importance

```python
feature_imp = dict(zip(clf.feature_names_in_, clf.feature_importances_))
feature_imp = {k: v for k,v in sorted(feature_imp.items(), key = lambda x: x[1], reverse=True)}
```

These insights into feature importance can help you understand which factors are most influential in predicting loan status. It's important to note that these percentages represent the relative importance of each feature in the model, and the actual impact of each feature may vary depending on the dataset and specific context.

## 16. Hyperparameter Tuning

```python
model_df
```

After careful evaluation, it is evident that the Logistic Regression model

 outperforms the other models with the highest accuracy score of 80.46%. Therefore, we have selected Logistic Regression as the best machine learning model for predicting loan status in this dataset.

Hence, we will employ GridSearchCV for hyperparameter tuning on the Logistic Regression model.

```python
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the parameter grid to search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Type of regularization
    'solver': ['liblinear', 'saga'],  # Solver algorithm
}

# Create the Logistic Regression model
logistic_regression = LogisticRegression(max_iter=1000)

# Create GridSearchCV with cross-validation
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')

# Fit the model to the data and perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best accuracy score
best_accuracy = grid_search.best_score_
print("Best Accuracy Score:", best_accuracy)
```

After performing hyperparameter tuning for the Logistic Regression model, the best hyperparameters and corresponding accuracy score are as follows:

Best Hyperparameters:

- C: 0.01
- penalty: 'l1'
- solver: 'liblinear'

Best Accuracy Score: 82.49%

These hyperparameters were determined through the grid search process using cross-validation, resulting in an improved accuracy score of 82.49% for the Logistic Regression model.

## 17. Predicting New Data Using The Best Model

We will employ the best-performing model, Logistic Regression, with the following hyperparameters: C: 0.01, penalty: 'l1', solver: 'liblinear', and max_iter: 1000, to make predictions on new data.

### Saving The Best Model

We are going to store the model in a binary format so that you can easily load it later for making predictions without the need to retrain the model. It is commonly used in machine learning to persist trained models.

```python
import joblib

logistic_regression = LogisticRegression(C = 0.01,
                                        penalty = 'l1',
                                        solver = 'liblinear',
                                        max_iter = 1000)
logistic_regression.fit(X_train, y_train)
joblib.dump(logistic_regression,'loan_status_predict')
best_model = joblib.load('loan_status_predict')
```

### Predicting using The Best Model

Suppose there is an individual who:

- The individual is male (Gender: Male), married (Married: Yes), has two dependents (Dependents: 2), holds a graduate degree (Education: Graduate), is not self-employed (Self_Employed: No), has an applicant income of 2889, no coapplicant income (CoapplicantIncome: 0.0), is applying for a loan amount of 45, with a loan term of 180 months (Loan_Amount_Term: 180), has no credit history (Credit_History: 0), and resides in a semiurban area (Property_Area: Semiurban).

Translating this using the label encoding, we want to predict:

```python
new_df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])
result = best_model.predict(new_df)
if result == 0:
    print("Loan is not approved")
else:
    print("Loan is approved")
```

Therefore, based on the best model, the loan is not approved for this specific person, with an accuracy of 82.49%.

## 18. Application-based Machine Learning Model

We're creating an application based on this machine learning model. You can input any values into the application, click the button, and within seconds, you'll receive the loan status determined by the best machine learning model, boasting an accuracy of 82.49%.

```python
from tkinter import *
import joblib
import pandas as pd

def show_entry():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())
    p9 = float(e9.get())
    p10 = float(e10.get())
    p11 = float(e11.get())
    
    model = joblib.load('loan_status_predict')
    df = pd.DataFrame({
        'Gender':p1,
        'Married':p2,
        'Dependents':p3,
        'Education':p4,
        'Self_Employed':p5,
        'ApplicantIncome':p6,
        'CoapplicantIncome':p7,
        'LoanAmount':p8,
        'Loan_Amount_Term':p9,
        'Credit_History':p10,
        'Property_Area':p11
    },index=[0])
    result = model.predict(df)
    
    if result == 1:
        Label(master, text="Loan approved").grid(row=31)
    else:
        Label(master, text="Loan Not Approved").grid(row=31)
    
master =Tk()
master.title("Loan Status Prediction Using Machine Learning")
label = Label(master,text = "Loan Status Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Gender [1:Male ,0:Female]").grid(row=1)
Label(master,text = "Married [1:Yes,0:No]").grid(row=2)
Label(master,text = "Dependents [1,2,3,4]").grid(row=3)
Label(master,text = "Education ['Graduate':1,'Not Graduate':0]").grid(row=4)
Label(master,text = "Self_Employed ['Yes':1,'No':0]").grid(row=5)
Label(master,text = "ApplicantIncome").grid(row=6)
Label(master,text = "CoapplicantIncome").grid(row=7)
Label(master,text = "LoanAmount").grid(row=8)
Label(master,text = "Loan_Amount_Term").grid(row=9)
Label(master,text = "Credit_History [Credit history meets guidelines ('Yes': 1, 'No': 0)]").grid(row=10)
Label(master,text = "Property_Area ['Rural':0,'Semiurban':1,'Urban':2]").grid(row=11)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)


e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)

Button(master, text='Predict', command=show_entry).grid(row=12, columnspan=2)

master.mainloop()
```

By running the above code, you can create a simple graphical user interface (GUI) application where you can input the applicant's information, click the "Predict" button, and the model will provide the loan status prediction based on the input data.
