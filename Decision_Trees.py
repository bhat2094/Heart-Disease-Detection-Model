# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Python >= 3
# pandas >= 0.25.1
# numpy >= 1.17.2
# sklearn >= 0.22.1

############################### Import Modules ###############################

import pandas as pd # to load and manipulate data
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs

from sklearn.tree import DecisionTreeClassifier # to build the classification tree
from sklearn.tree import plot_tree # to draw a classification tree

from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.model_selection import cross_val_score # for cross validation

# from sklearn.metrics import confusion_matrix # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix

################################ Import Data #################################

# now pandas reads the data into a dataframe (df). The function in pandas to read csv files is read_csv
# df = pd.read_csv('processed.cleveland.data', header=None)

# reading the data directly from the machine learning repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)

print(' ')
print('Printing the raw data:')
print(df.head()) # to check the first 5 rows of data
# we can see there is not column names which makes it heard to read the data.
# Here, we are giving names to the columns

df.columns = ['age',
              'sex',
              'cp', # chest pain
              'restbp', # resting blood pressure (mm Hg)
              'chol', # serum cholesterol (mg/dl)
              'fbs', # fasting blood sugar
              'restecg', # resting electrocardiographic results
              'thalach', # maximum heart rate achieve
              'exang', # exercise induced angina
              'oldpeak', # ST depression induced by exercise relative to rest
              'slope', # the slope of the peak exercise ST segment
              'ca', # number of major vessels (0-3) colored by fluoroscopy
              'thal', # short for thalium heart scan
              'hd', # diagonisis of heart disease <<< predicted attribute
              ]

print(' ')
print('Printing the data after giving column name:')
print(df.head()) # to check the first 5 rows of data again

######################### Identifying Missing Data ###########################

print(' ')
print('Checking the data type for each column:')
print(df.dtypes) # to check the data type for each column, such as float64, int64, etc.

# If the data type is "object" that usually means there is a mixture of letters and numbers.
# Since "ca" and "thal" columns have data type "object", we need to further investigate these columns.

print(' ')
print('Finding the unique values from the "ca" column:')
print(df['ca'].unique()) # prints unique values from the "ca" column
print(' ')
print('Finding the unique values from the "thal" column:')
print(df['thal'].unique()) # prints unique values from the "thal" column

# the question mark in the unique value represent missing data

######################### Dealing with Missing Data ##########################

# we are using scikit-learn classification tree. Since it does not support 
# missing values, we need to either get rid of the missing value columns or 
# make an educated guess.

# we are going to count the number fo rows that contains missing values
# loc[], short for 'location', lets us specify which rows we want
# we want any row with '?' in column 'ca' OR any row with '?' in column 'thal'
# len(), short for 'length', prints out the number fo rows
print(' ')
print('Total number of missing rows in "ca" and "thal" columns:')
print(len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')]))

# If the number of missing values are not too high, we can just print them out
# and look at them
print(' ')
print(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

# now we want to see the number of rows in the entire dataframe
print(' ')
print('Total number of rows in the raw dataset:')
print(len(df))

# if the number of rows with missing values is few % of the total number of rows,
# we can just simply remove the rows without compromising the data

# In our case, we have 6 rows with missing values among 303 rows which is 2 % 
# of the data. So, we are going to remove these rows with missing values.

df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]
print(' ')
print('Total number of rows in the dataset after removing rows with missing values:')
print(len(df_no_missing))

# just to verify there is no missing values in 'ca' or 'thal' anymore
print(' ')
print('Finding the unique values from the "ca" column after removing missing rows:')
print(df_no_missing['ca'].unique()) # prints unique values from the "ca" column
print(' ')
print('Finding the unique values from the "thal" column after removing missing rows:')
print(df_no_missing['thal'].unique()) # prints unique values from the "thal" column
# if there is no '?', we are good to go

############ Split Data into Dependent and Independent Variables #############

# split data into two parts:
# 1. columns of data we will use to make prediction (everything but the last column, hd)
# 2. colum of data that we want to predict (the last column, hd or heart disease)

# creating the independent variable by copying all the columns from 
# df_no_missing except the 'hd' column 
X = df_no_missing.drop('hd', axis=1).copy()

# creating the dependent variable
y = df_no_missing['hd'].copy()

# we can print few rows of these variable to verify
# now we need to format data suitable for ddecision tree model

############################# Formatting the Data ############################

# we are going to look at the data type for each attribute in the variable X
# age -- float
# sex -- category
#        0 = female
#        1 = male
# cp, chest pain -- category
#                   1 = typical angina
#                   2 = atypical angina
#                   3 = non-angina pain
#                   4 = asymptomatic
# restbp, resting blood pressure -- float
# chol, serum cholesterol -- float
# fbs, fasting blood sugar -- category
#                             0 = >= 120 mg/dl
#                             1 = < 120 mg/dl
# restecg, resting electrocardiographic results -- category
#                                                  1 = normal
#                                                  2 = having ST-T wave abnormality
#                                                  3 = showing probable or definite left ventricular hypertrophy
# thalach, maximum heart rate achieve -- float
# exang, exercise induced angina -- category
#                                   0 = no
#                                   1 = yes
# oldpeak, ST depression induced by exercise relative to rest -- float
# slope, the slope of the peak exercise ST segment -- category
#                                                     1 = upsloping
#                                                     2 = flat
#                                                     3 = downsloping
# ca, number of major vessels (0-3) colored by fluoroscopy -- float
# thal, short for thalium heart scan -- category
#                                       3 = normal (no cold spots)
#                                       6 = fixed defect (cold spots during rest and exercise)
#                                       7 = reversible defect (cold spots only appear during exercise)

# print(' ')
# print(X.dtypes)

# we can see that most of the datatypes in Python is not matching with the 
# actual data type listed above

# scikit-learn decision tree support continious data, but not categorical data.
# to use catagorical data in scikit-learn, we need to convert catagorical data 
# into multiple binary columns
# we can convert catagorical data into a series of binary columns with 
# 0's and 1's using One-Hot encoding
# There are two common ways/function of On-Hot encoding in Python
# ColumnTransformer() -- in scikit-learn
# get_dummies() -- in pandas

# to check what 'get_dummies()' does, we can use the following
# pd.get_dummies(X, columns=['cp']).head()

# we are going to use On-Hot encoding on the attributes with mre than 2 catagories
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])
# print(' ')
# print(X_encoded.head())

# since the rest of the attributes in X has two catagories and the values are 
# only 1's and 0's, we do not need to do anything for those catagorical attributes

# moving on to the dependent variable, y
print(' ')
print('raw y values:')
print(y.unique())
# y = 0 means no heard disease and the rest of the values represent the strentgh
# of heart disease. y = 4 is the most severe heart disease. For this 
# relatively simple classifier we will just predict if the person has (y = 1) 
# or does not have (y = 0) heart disease.
y_not_zero_indices = y > 0 # index for non-zero elements in y
y[y_not_zero_indices] = 1 # set each non-zero values in y to 1
print(' ')
print('y values for our model:')
print(y.unique())

# now the data is properly formated to make a classification tree

################### Building the Decision Tree Classifier ####################

# split our data into traiing and testing sets
# we defined the random_state to be 42 so that we split the data exactly the 
# same way everytime we run the program
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

# create the decision tree
clf_dt = DecisionTreeClassifier(random_state=42)
# fit the tree to the training data
clf_dt = clf_dt.fit(X_train, y_train)

# we can plot the tree and visualize the training classification
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, # passing the tree handle
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns
          );

# after training the tree, we are going to use the testing data and plot the 
# confusion matrix
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["Does not have HD", "Has HD"])

# we can calculate the true positive, true negative, false positive, and false 
# negative from the confusion matrix and evaluate the performance of the model

# in this case, true positive = 74% and true negative is 79%. looking at the 
# tree itself, it seems that the tree is very complex and might have overfit 
# the data. in general, decision trees are prone to overfitting because of the
# large number of parameters in the tree model

# we can prune the tree to minimize overfitting and optimize the model
# 1. Cost Complexity Pruning
# 2. Cross Validation

############### Model Optimization: Cost Complexity Pruning ##################

# we need to find the right value of the puring parameter, alpha. we can plot
# alpha as a function of the accuracy of the tree for both the training and 
# testing data to find the optimum alpha.

path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # etermine value for alpha
ccp_alphas = path.ccp_alphas # extract different values for alphas
ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha. it corresponds to the root node

clf_dts = [] # create an empty array to put decission trees

# create a decision tree for each value of alpha and store it in clf_dts
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha = ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

# plotting the accuracy as a function of alpha for the training set and the 
# testing set
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Alpha for Training and Testing Sets")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = 'train', drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker = 'o', label = 'test', drawstyle = "steps-post")
ax.legend()
plt.show()

# as we increase alpha, the size of the tree gets smaller, the accuracy of the
# training set decreases, but more importantly the accuracy of the testing
# set gets better. just by looking at the plot we can say the optimized value
# for alpha is about 0.016. 

# classification tree -- the value of alpha goes from 0 to 1 because gini 
# score goes from 0 to 1
# regressing tree -- alpha can be from 0 to +infinity

################### Model Optimization: Cross Validation #####################

# the optimized value of alpha of 0.016 from the Cost Complexity Pruning method
# is only true for the way we split the training and testing data sets at the
# begining. if we split the data in some other way, the optimized alpha can be 
# different. using cross validation we can find the best value for alpha for 
# the given dataset irrespective of the spliting.

# creating a tree with ccp_alpha=0.016
clf_dt = DecisionTreeClassifier(random_state=43, ccp_alpha=0.016)

# now 5-fold cross validation creates 5 sets of training and testing data
# we can use these 5 sets to train and test the tree
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})

df.plot(x='tree', y='accuracy', marker='o', linestyle='--')
# the plot shows that for different splits of data for a given alpha results in
# different accuracy. so, we want to optimize the data spliting

# we will use cross validation to get the optimum Cost Complexity Pruning alpha

alpha_loop_values = [] # to store the result for each fold during cross validation

# create a decision tree for each value of alpha
# for each candidate of alpha, we will run 5-fold cross validation
# we store the mean and the standard deviation of the scores (the accuracy) 
# for each call to cross_val_score to alpha_loop_values
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha = ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# plot the means and standard deviation of the scores for each alpha
alpha_results = pd.DataFrame(alpha_loop_values, 
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--'
                   )

# from the plot we can see that the optimized alpha for the dataset is around 0.014
# we can find the exact value as follows
# print(' ')
# print('Optimized alpha for the dataset using cross validation')
# print(alpha_results[(alpha_results['alpha'] > 0.014) 
#                     & 
#                     (alpha_results['alpha'] < 0.015)])

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014) 
                               & 
                               (alpha_results['alpha'] < 0.015)]['alpha']

# converting ideal_ccp_alpha from a series to a float
ideal_ccp_alpha = float(ideal_ccp_alpha)
print(' ')
print('Optimized alpha for the dataset using cross validation')
print(ideal_ccp_alpha)

# now we have the alpha that we need for our dataset. we can built, evaluate,
# draw, and interpret the final classification tree

###################### Final Classification Tree Building ####################

# building the final decision tree with optimized alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42,
                                       ccp_alpha = ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

# plotting confussion matrix for the final decision tree
plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=["Does not have HD", "Has HD"])

# plotting the pruned tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          filled = True,
          rounded = True,
          class_names = ["No HD", "Yes HD"],
          feature_names=X_encoded.columns
          );
