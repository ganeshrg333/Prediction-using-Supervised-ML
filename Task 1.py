#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# ## Task 1-: Prediction using Supervised ML
# 
# ## Name of Author-: Ganesh Ravindra Gaonkar
# 
# 
# 

# ### Step 1-: Import necessary libraries

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics  


# ### Step 2-: Import the dataset

# In[5]:


dataset = pd.read_csv("Downloads\Supervised_ML.csv")


# In[6]:


dataset


# ### Step 3-: The first five values in the dataset
# 
# 

# In[7]:


dataset.head()


# ### Step 4-: Describe the dataset

# In[8]:


dataset.describe()


# ### Step 5-: Visualization of dataset

# In[9]:


plt.scatter(dataset['Hours'], dataset['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Studied Hours')
plt.ylabel('Scores')
plt.show()


# ### Step 6-: Train-Test Split

# In[10]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ### Step 7-: Training the Simple Linear Regression model on the Training set

# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ### Step 8-: Plotting the regression line

# In[12]:


line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line,color = 'green');
plt.show()


# ### Step 9-: Predicting the Test set results

# In[13]:


y_pred = regressor.predict(X_test)
print(y_pred)


# ### Step 10-: Visualising the Training set results

# In[14]:


plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Hours vs. Percentage (Training set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# ### Step 11-: Comparing the actual values with the predicted ones

# In[15]:


dataset = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
dataset


# ### Step 12-: predicting the score 

# In[16]:


dataset = np.array(9.25)
dataset = dataset.reshape(-1, 1)
pred = regressor.predict(dataset)
print("If the student studies for 9.25 hours/day, the score is {}.".format(pred))


# ### Step 13-: Evaluating the Model

# In[19]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# ### Conclusion-: 
# 
# ### According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 92.91 marks.
