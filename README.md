### Credit Card Fraud Detection

### Abstract

The project aims to classify the fraudulent transactions of credit cards, with high accuracy. The Credit Card Fraud Detection problem includes modeling credit card transactions with the knowledge of the ones that turns out to be fraud. Such modeling will enable companies to efficiently flag a transaction which may be fraudulent. 

### The dataset 
The dataset consists of simulated credit card transactions containing both legitimate and fraudulent transactions. The dataset is in csv format and has been taken from the Kaggle public dataset. Data has been generated using Sparkov Data Generation tool created by Brandon Harris (Duration-Jan 1, 2019 to Dec 31, 2020). 
  - The data consists of around 1.3 million transactions having details of 1000 customers with a pool of 800 merchants. 
  - It consists of 21 features. Some of the important features are cc_num, merchant, category, amount, gender, city_population, job, date of birth, is_fraud, city, state,zip etc.

### Goal 
To predict the fraudulent transaction with 100% accuracy.

Steps followed in notebook :
### 1. Loading the data 

```ruby
df_train = pd.read_csv("fraudTrain.csv")
df_test = pd.read_csv("fraudTest.csv")
```
### 2. Data Exploration
#### 2.1 Data Information
```ruby
df.info()
```
#### 2.2 Shape of Data


```ruby
print(df_train.shape,df_test.shape)
df_train.isnull().sum()
df_train.corr()
```
#### 2.3 Taking a fraction to run the model faster
```ruby
# taking smaller sample to run the model faster
df_train= data_train.sample(frac = 0.1,random_state=1)
df_test= data_test.sample(frac = 0.05,random_state=1)
print(df_train.shape,df_test.shape)
```
#### 2.4 Checking the null values
```ruby
df_train.isnull().sum()
df_test.isnull().sum()
```
#### 2.5 Correlation Matrix
```ruby
#get correlations of each features in dataset
corrmat = df_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))

#plot heat map
g=sns.heatmap(df_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
```
#### 2.6 Histograms
```ruby
#visual representation of the data using histograms 
df_train.hist(figsize = (15, 15))
plt.show()
```
#### 2.7  Getting the Fraud and the Normal  transaction numbers for test and train datase
```ruby
fraud_train = df_train[df_train['is_fraud']==1]
normal_train = df_train[df_train['is_fraud']==0]
fraud_test = df_test[df_test['is_fraud']==1]
normal_test = df_test[df_test['is_fraud']==0]

print("Normal cases in train set :",len(df_train)-len(fraud_train),"\nFraud cases in train set :",len(fraud_train))
print("Normal cases in test set :",len(df_test)-len(fraud_test),"\nFraud cases in test set :",len(fraud_test))
```

### 3. Data transformation and feature engineering

From the exploratory data Analysis we make the following observations:   

1. The first column contains just the indices and is not useful so we will drop it.   
2. The third column with customer card number is also not useful , so we will drop it. 
3. "first name" and "last name" can also be dropped.  
4. Transaction number - is it really needed? can be dropped.
5. Date time column can be used to calculate the age of the customer

#### 3.1 Dropping the columns not needed
```ruby
# function to drop tbe columns
def dropCol(data):
    col_to_drop = ['trans_date_trans_time','Unnamed: 0','cc_num','first','last','trans_num']
    res = data.drop(col_to_drop,axis = 1)
    return res
```
```ruby
# dropping the columns
# dropping the columns ['trans_date_trans_time','Unnamed: 0','cc_num','first','last','trans_num']
# train data set
df_train = dropCol(df_train)
# test data set
df_test = dropCol(df_test)

print ( df_train.shape, df_test.shape)
```
#### 3.2 Creating Independent and Dependent Features
```ruby

# Create independent and Dependent Features
columns = train_sample.columns.tolist()

# removing the dependent feature is_fraud
columns = [c for c in columns if c not in ["is_fraud"]]
X_train = train_sample[columns]
Y_train = train_sample['is_fraud']
X_test = df_test[columns]
Y_test = df_test['is_fraud']
print ( X_train.shape, Y_train.shape,X_test.shape, Y_test.shape)
```


#### 3.3 Convering the date of birth to age

```ruby
# function to convert dob to years
def age_years(born):
    return 2019 - int(born[0:4])

# replacing the dob column with age column in our data set for test and train
X_train['age'] = X_train['dob'].apply(lambda x: age_years(x))
X_train = X_train.drop(['dob'],axis =1)

X_test['age'] = X_test['dob'].apply(lambda x: age_years(x))
X_test = X_test.drop(['dob'],axis =1)
print(X_train.shape,X_test.shape)

```

#### 3.4 Converting the categorical features to numerical by one- hot - encoding
```ruby
# concanating the test and train data so that number of columns remain the same in both the data sets
final_df=pd.concat([X_train,X_test],axis=0)
final_df.shape

```
```ruby
# creating the list of categorical variables
categorical_features =[feature for feature in X_train.columns if final_df[feature].dtypes == 'O']
categorical_features

```
```ruby
#observing the unique values in each feature
for feature in categorical_features:
    print("Distinct categories for {}  are {}".format(feature,len(final_df[feature].unique())))

```
```ruby
# function to convert categorical variables to one hot encoding
def category_onehot_multcols(data,multcolumns):
    df_final = data
    i=0
    for fields in multcolumns:
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:           
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1             
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final``ruby

```
```ruby
# applying the one hot encoding
final_df = category_onehot_multcols(final_df, categorical_features)

```
```ruby
# removing duplicated columns
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape

```
```ruby
# separating the test and training data
df_Train=final_df.iloc[:129668,:]
df_Test=final_df.iloc[129668:,:]
print(df_Train.shape,df_Test.shape)

```
```ruby
# files ready for testing on models
print(df_Train.shape, df_Test.shape, Y_train.shape, Y_test.shape)
```
As we see that the data is skewed, i.e the number of samples for class 1 is less tha 0.5% of the samples of class 0.
In this case the the machine learning algorithms with not predict the Fraud cases correctly.
To predict the Fraud correctly we have two approaches:   
1. Handle the imbalance in Data and apply various Machine Learning algorithms    
2. Predict Fraud as outlier/Anomaly  

We proceed for the first method first.


### 4. Handling the imbalance in dataset.

There can be two approaches to predict the the fraud cases:   
Handling the Imbalance in data by one of the following methods:   
> 4.1 Under Sampling   
       4.2 Over Sampling   
       4.3 SMOTE (Synthetic Minority over sampling technique)   
       4.4 Near Miss algorighm ( under sampling )   
       4.5 Ensemble method 
   
First we apply the model without handling the imbalance. We will use Logistic Regression and Decision Tree classifier for our exploration of the method to apply to handle imbalance in data. 
**Logistic Regression**
```ruby   
# Logistic Regression
model_LR = LogisticRegression()
model_LR.fit(df_Train,Y_train)
y_pred = model_LR.predict(df_Test)

```
```ruby
#Let's evaluate our model 
def print_eval(y_pred,model):
    print("Training Accuracy: ",model.score(df_Train, Y_train))
    print("Testing Accuracy: ", model.score(df_Test, Y_test))
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    print(classification_report(Y_test,y_pred))
print_eval(y_pred,model_LR)  
```
 **Decision Tree**
 ```ruby
 # decision tree
decision_tree_model = DecisionTreeClassifier(random_state=137)
decision_tree_model.fit(df_Train,Y_train)
y_pred = decision_tree_model.predict(df_Test)
print_eval(y_pred,decision_tree_model)
 ```
#### Metric Analysis

1. **True  Positives** : Correctly classified as Safe Transaction     = 27333 (.99)   
2. **False Negitives** : Mis-classified Safe Transaction              = 323 (Harmless)   
3. **False Positives** : Mis-classified as Fraud Transactions         = 47 (Dangerous )   
4. **True Negatives**  : Correctly classified as Fraud Transactions   = 83 (out of 130 -> .64)   
5. **Accuracy**        : .99   

Here the accuracy is not taken into account as it is misleading.   
We want to get maximum True Negatives i.e we want to predict the Fraud tranactions with maximum accuracy. This can be done by monitoring the Recall.

So when the data is imbalanced the Recall is    
class 0 (Safe) : 0.99   
class 1 (Fraud): 0.64   

**We want the recall of class 1 to be close to 1.00**    


#### 4.1.1 Under Sampling
```ruby
# adding the dependent feature in the train data set
print(Y_train.shape,df_Train.shape)
df_train = pd.concat([df_Train,Y_train],axis = 1)
df_train.shape

```

```ruby
# Class count
count_class_0, count_class_1 = df_train.is_fraud.value_counts()
print(count_class_0, count_class_1)
# Divide by class
df_class_0 = df_train[df_train['is_fraud'] == 0]
df_class_1 = df_train[df_train['is_fraud'] == 1]
print(df_class_0.shape,df_class_1.shape)

```
```ruby
# Undersample 0-class and concat the DataFrames of both class
df_class_0_under = df_class_0.sample(count_class_1)
df_train_under_sample = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_train_under_sample.is_fraud.value_counts())

```
```ruby
# training and predictions : Logistic Regression
X = df_train_under_sample.drop('is_fraud',axis='columns')
y = df_train_under_sample['is_fraud']
model_LR_under_sample = LogisticRegression()
model_LR_under_sample.fit(X,y)
y_pred = model_LR.predict(df_Test)
print_eval(y_pred,model_LR_under_sample)
```
```ruby
# training and predictions : decision tree
decision_tree_model = DecisionTreeClassifier(random_state=137)
decision_tree_model.fit(X,y)
y_pred = decision_tree_model.predict(df_Test)
print_eval(y_pred,model_LR_under_sample)

```

#### 4.1,2 Over Sampling
```ruby
# Class count
count_class_0, count_class_1 = df_train.is_fraud.value_counts()
print(count_class_0, count_class_1)
# Divide by class
df_class_0 = df_train[df_train['is_fraud'] == 0]
df_class_1 = df_train[df_train['is_fraud'] == 1]
print(df_class_0.shape,df_class_1.shape)
```
```ruby
# Oversample 1-class and concat the DataFrames of both class
df_class_1_over = df_class_1.sample(count_class_0,replace=True)
df_train_over_sample = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')

print(df_train_over_sample.is_fraud.value_counts())
```
```ruby
# training and predictions : Logistic Regression
X = df_train_over_sample.drop('is_fraud',axis='columns')
y = df_train_over_sample['is_fraud']
model_LR_over_sample = LogisticRegression()
model_LR_over_sample.fit(X,y)
y_pred = model_LR_over_sample.predict(df_Test)
print_eval(y_pred,model_LR_over_sample)
```
```ruby
# training and predictions : decision tree
decision_tree_model_over_sample = DecisionTreeClassifier(random_state=137)
decision_tree_model_over_sample.fit(X,y)
y_pred = decision_tree_model_over_sample.predict(df_Test)
print_eval(y_pred,decision_tree_model_over_sample)
```

#### 4.3 Implementing SMOTE (Synthetic Minority Oversampling Technique)
```ruby
# smote implementation
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(X, y)
y_sm.value_counts()
```
```ruby
# training and predictions : Logistic Regression

model_LR_smote = LogisticRegression()
model_LR_smote.fit(X_sm,y_sm)
y_predict = model_LR_smote.predict(df_Test)
print_eval(y_pred,model_LR_smote)
```
```ruby
# training and predictions : decision tree
decision_tree_model_smote = DecisionTreeClassifier(random_state=137)
decision_tree_model_smote.fit(X_sm,y_sm)
y_pred = decision_tree_model_smote.predict(df_Test)
print_eval(y_pred,decision_tree_model_smote)
```

#### 4.4 Near Miss (NearMiss Algorithm – Undersampling)
```ruby
# near miss
nr = NearMiss() 
X_train_miss, y_train_miss = nr.fit_sample(X, y) 
print('Near Miss:')
print(y_train_miss.value_counts())
```
```ruby
# training and predictions : Logistic Regression
model_LR_smote = LogisticRegression()
model_LR_smote.fit(X_train_miss,y_train_miss)
y_predict = model_LR_smote.predict(df_Test)
print(classification_report(Y_test, y_predict))

```
```ruby
# training and predictions : decision tree
decision_tree_model_nm = DecisionTreeClassifier(random_state=137)
decision_tree_model_nm.fit(X_train_miss,y_train_miss)
y_pred = decision_tree_model_nm.predict(df_Test)
print_eval(y_pred,decision_tree_model_nm)
```
### 4.5 Comparisions of different models
```ruby
# Test models
classdict = {'normal':0, 'fraudulent':1}
print()
print('========================== Model Test Results ========================' "\n")   
modlist = [('dc', decision_tree_model),
           ('dc_us', decision_tree_model_undersample),
           ('dc_os', decision_tree_model_over_sample),
           ('dc_smote', decision_tree_model_smote),
           ('dc_nm', decision_tree_model_nm)
           
          ] 
models = [j for j in modlist]
for i, v in models:
    accuracy = metrics.accuracy_score(Y_test, v.predict(df_Test))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(df_Test))
    classification = metrics.classification_report(Y_test, v.predict(df_Test))   
    print('=== {} ==='.format(i))
    print ("Model Accuracy: ",  '{}%'.format(np.round(accuracy, 3) * 100))
    print()
    print("Recall:" "\n", confusion_matrix[1][1]/130)
    print()
    #pf.plot_confusion_matrix(confusion_matrix, classes = list(classdict.keys()),title='Confusion Matrix Plot', cmap=plt.cm.summer)
    print() 
    print("Classification Report:" "\n", classification) 
    print() 

print('============================= ROC Curve ===============================' "\n")      
pf.plot_roc_auc(arg1=models, arg2=df_Test, arg3=Y_test)


```
```ruby

```

### 5. Model Implementation
```ruby

```
```ruby

```
```ruby

```
```ruby

```
```ruby

```
```ruby

```
```ruby

```
### 6. Outlier Detection Techniques

### 7. Predictions
### 8. Results


