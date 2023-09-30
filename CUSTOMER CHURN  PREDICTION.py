#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libariries
import numpy as np
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


# Uploading the dataset in a dataframe
df=pd.read_csv("Churn_Modelling.csv")
df.head(10)


# In[4]:


df.info()


# In[3]:


#Counting the label value
df['Exited'].value_counts()


# In[4]:


# Visualization of how the features are related 
relations=["Gender","NumOfProducts","HasCrCard","IsActiveMember"]
numerical=relations
plt.figure(figsize=(20,4))

for i, relation in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sb.countplot(x=str(relation), data=df)
    ax.set_title(f"{relation}")


# In[5]:


#Create a mapping of gender categories to numeric values
gender_mapping = {'Male': 0, 'Female': 1}
df['Gender'] = df['Gender'].map(gender_mapping)


# In[6]:


df.info()


# In[7]:


# Drop the unnecassary columns
df = df.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)


# In[8]:


# Perform one-hot encoding for 'Geography' column
df = pd.get_dummies(df, columns=['Geography'], prefix=['Geography'])


# In[ ]:





# In[9]:


df.head()


# In[13]:


df.info()


# In[10]:


# Scaling the column 
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

columns_to_scale=['Balance','EstimatedSalary']

df[columns_to_scale]=scaler.fit_transform(df[columns_to_scale])


# In[11]:


# Training testing split of data
from sklearn.model_selection import train_test_split

X = df.drop('Exited', axis=1)  # Features
y = df['Exited']  # Target variable

# Split the data into training and testing sets (e.g., 80% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[12]:


#Imoorting RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

model.fit(X_train,y_train)


# In[13]:


#Evalution of model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Print or use the evaluation metrics as needed
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", confusion)


# In[ ]:





# In[ ]:





# In[ ]:




