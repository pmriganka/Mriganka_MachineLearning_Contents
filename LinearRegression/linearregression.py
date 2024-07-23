# %%
'''
IMPORTING THE LIBRARIES
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
'''
IMPORTING THE DATASETS
'''
dataset = pd.read_csv('C:\\Users\\pmrig\\OneDrive\\Desktop\\MyLearnings\\MachineLearning\\Mriganka_MachineLearning_Contents\\LinearRegression\\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %%
'''
SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
X_train - independent variable
y_train - dependent variable
random_State random split in data
'''
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)  

# %%
'''
TRAINING the SIMPLE LINEAR REGRESSION MODEL on the TRAINING SET

fit is a method of the Linear Regression Class to train the training set
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %%
'''
PREDICTING TEST RESULT
'''
y_pred = regressor.predict(X_test)

# %%
'''
Visualize the Training Set Results
'''
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %%
'''
Visualize the Test Set Results
'''
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %%
'''
The salary of an employee with 12 years of experience
Notice that the value of the feature (12 years) was input in a double pair of square brackets. 
That's because the "predict" method always expects a 2D array as the format of its inputs. 
And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:
12→scalar 
[12]→1D array 
[[12]]→2D array 
'''
print(regressor.predict([[12]]))


# %%
'''
Getting the final linear regression equation with the values of the coefficients
Therefore, the equation of our simple linear regression model is:
Salary=9345.94×YearsExperience+26816.19 
Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object.
Attributes in Python are different than methods and usually return a simple value or an array of values.
'''
print(regressor.coef_)
print(regressor.intercept_)


# %%
dataset.columns
X = dataset[['YearsExperience']]
cdf = pd.DataFrame(regressor.coef_, X.columns, columns=['Coeff'])

# %%
'''
For 1 year experience increase salary increase by 9440 unit
'''
cdf


