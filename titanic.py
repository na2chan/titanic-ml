#%%
# Data Analysis
import pandas as pd
import numpy as np
import random as rnd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import SGDClassifier

# Read in Training and Test Data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Drop meaningless features
df_train = df_train.drop(['Name', 'Cabin', 'Ticket'], axis=1)
df_test = df_test.drop(['Name', 'Cabin', 'Ticket'], axis=1)

#%% 
# Data Visualization

# NULL Heatmap - indicates which features have NULL values
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)
# the heatmap suggests we should make some substitution for the NULL values 
# of the Age covariate as the number of NULL observations are limited

#%%
# histogram of Sex to Survival
sns.countplot(x='Survived',hue='Sex',data=df_train,palette='afmhot')

#%%
# histogram of Passenger Class to Survival
sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='hsv')

#%%
# boxplot of Pclass + Age ~ Survival
sns.boxplot(x='Pclass',y='Age',data=df_train)
# Mean Age for each Passenger Class
df_train.groupby('Pclass', as_index=False)['Age'].mean()

#%%
# boxplot of Sex + Age ~ Survival
sns.boxplot(x='Sex',y='Age',data=df_train)
# Mean Age for each Sex
df_train.groupby('Sex', as_index=False)['Age'].mean()

#%%
plt.show()

#%%
# These plots suggest that Age has a strong relationship with Sex and Passenger Class
# in terms of Survival Rate
# We can substitute NULL values in Age with the mean across Sex and Passenger Class
means = df_train.groupby(['Sex','Pclass'])['Age'].mean()

def substituteNULL(column):
    Age = column[0]
    Sex = column[1]
    Pclass = column[2]

    if pd.isnull(Age):
        if Sex == 'male' and Pclass==1:
            return 41
        elif Sex == 'male' and Pclass==2:
            return 31
        elif Sex == 'male' and Pclass==3:
            return 26
        elif Sex == 'female' and Pclass==1:
            return 35
        elif Sex == 'female' and Pclass==2:
            return 29
        else:
            return 22
    else:
        return Age

df_train['Age'] = df_train[['Age','Sex','Pclass']].apply(substituteNULL,axis=1)
df_test['Age'] = df_train[['Age','Sex','Pclass']].apply(substituteNULL,axis=1)

#%%
# Also need to substitute the NULL fare in df_test
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)

#%%
#Since we have category variables, we can one-hot encode the binary ones/use indicator variables
    # For training
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
df_train.drop(['Sex','Embarked'],axis=1,inplace=True)
df_train = pd.concat([df_train,sex,embark],axis=1)

    # For Test
sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)
df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
df_test = pd.concat([df_test,sex,embark],axis=1)

    # Also need to drop passengerID as has no importance in the actual training process
df_train.drop(['PassengerId'],axis=1,inplace=True)
Passenger_ID = df_test['PassengerId'] # Saving for later
df_test.drop(['PassengerId'],axis=1,inplace=True)


#%%
from sklearn.model_selection import train_test_split

x = df_train.drop('Survived', axis = 1)
y = df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1),df_train['Survived'], test_size = 0.25,random_state=100)

#%% 
# Train model using stochastic gradient descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
print(acc_sgd)

# %%
# Run model on test data
survived = sgd.predict(df_test)
df_test['Survived']=survived
df_test['PassengerID']=Passenger_ID

# %%
# Write results to CSV
df_test[['PassengerID', 'Survived']].to_csv('Titanic_LogRegression.csv', index=False)