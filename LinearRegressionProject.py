
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


# In[2]:


os.chdir("C:/Users/rusha/Desktop/datasets/")


# In[3]:


#Load the csv
df = pd.read_csv("train.csv")


# In[4]:


#The correlation matrix
correlation_values = df.select_dtypes(include=[np.number]).corr()


# In[5]:


correlation_values


# In[28]:


#Select the features that have a correlation coeff between -0.6 to 0.6 with the target variable
selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]
#Correlation between the selected variables
correlation_values.loc[selected_features.index,selected_features.index]


# In[60]:


#Train test split
X = df[["OverallQual","TotalBsmtSF","GrLivArea","GarageArea"]]
y = df["SalePrice"]
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.315,random_state=42)


# In[61]:


#Fitting the Linear Regression Model
regressor = LinearRegression(normalize=True)
regressor.fit(X_train,y_train)


# In[62]:


#Test the model 
y_prediction = regressor.predict(X_test)
#Validate the model
r_square = regressor.score(X_test,y_test)
rmse = np.sqrt(mean_squared_error(y_test,y_prediction))
print(r_square,rmse)


# In[68]:




