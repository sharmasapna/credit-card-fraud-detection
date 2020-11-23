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
```ruby
print(df_train.shape,df_test.shape)
df_train.isnull().sum()
df_train.corr()
```
```ruby
## Correlation
import seaborn as sns

#get correlations of each features in dataset
corrmat = df_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(df_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
```
```ruby
# plotting the classes
count_classes = pd.value_counts(df_train['is_fraud'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")
LABELS = ['Normal','Fraud']
plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
```

### 3. Data transformation and feature engineering

#### 3.1 Since the data set is large we wil work on a fraction of the data set to save time
```ruby
# sampling to run the model faster

train_sample= df_train.sample(frac = 0.1,random_state=1)
train_sample.shape

```
Observing the data after sampling

```ruby
# observing the data after sampling
fraud_sample = train_sample[train_sample['is_fraud']==1]
valid_sample = train_sample[train_sample['is_fraud']==0]

fraud_org = df_train[df_train['is_fraud']==1]
valid_org = df_train[df_train['is_fraud']==0]
# sanity check
outlier_fraction = len(fraud_org)/float(len(valid_org))
outlier_fraction_sample = len(fraud_sample)/float(len(valid_sample))

print("Outlier fraction original:{} \nOutlier fraction Sample  :{}".format(outlier_fraction,outlier_fraction_sample))
#print("Fraud Cases Sample : {}".format(len(fraud_sample)))
#print("Valid Cases Sample : {}".format(len(valid_sample)))
```
#### 3.2 Dropping the columns

The first column contains just the indices and is not useful so we will drop it.
The third column with customer card number is also not useful , so we will drop it. 
First name and Last name can also be dropped.
Transaction number - is it really needed? can be dropped.

```ruby
def dropCol(data):
    col_to_drop = ['trans_date_trans_time','Unnamed: 0','cc_num','first','last','trans_num']
    res = data.drop(col_to_drop,axis = 1)
    return res
new = dropCol(df_train)
```


```ruby
# dropping the columns
# dropping the columns ['trans_date_trans_time','Unnamed: 0','cc_num','first','last','trans_num']
# complete data set
train_sample = dropCol(train_sample)
# fraud
fraud_sample = dropCol(fraud_sample)
#valid
valid_sample = dropCol(valid_sample)
# for test data
X_test = dropCol(df_test)
print ( train_sample.shape, fraud_sample.shape, valid_sample.shape,X_test.shape)
```
#### 3.3 Creating Independent and Dependent Features
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

#### 3.4 Feature engineering
##### 3.4.1 Convering the date of birth to age

```ruby
import numpy as np
import datetime
from datetime import date
def age_years(born):
    return 2019 - int(born[0:4])

X_train['age'] = X_train['dob'].apply(lambda x: age_years(x))
X_train = X_train.drop(['dob'],axis =1)

X_test['age'] = X_test['dob'].apply(lambda x: age_years(x))
X_test = X_test.drop(['dob'],axis =1)
print(X_train.shape,X_test.shape)
```

```ruby

```
```ruby

```
```ruby

```

### 4. Handling the imbalance in dataset.
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
### 6. Predictions
### 7. Results


