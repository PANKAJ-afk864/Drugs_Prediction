#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import files
uploaded = files.upload()


# In[3]:


# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# In[5]:


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# In[6]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[7]:


# 2. Load and Filter Data
df = pd.read_excel("drugsCom_raw.xlsx")
target_conditions = ["Depression", "High Blood Pressure", "Diabetes, Type 2"]
df = df[df["condition"].isin(target_conditions)].copy()
df.dropna(subset=['review', 'rating'], inplace=True)
df.drop_duplicates(inplace=True)


# In[8]:


# 3. EDA
print("\n--- EDA ---")
print(df['condition'].value_counts())
print(df['rating'].describe())

sns.countplot(data=df, x='condition')
plt.title('Review Count by Condition')
plt.xticks(rotation=45)
plt.show()

sns.histplot(data=df, x='rating', bins=10, kde=True)
plt.title('Rating Distribution')
plt.show()


# In[27]:


# 4. Preprocess Text and Create Sentiment Labels

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

df['clean_review'] = df['review'].apply(preprocess_text)

# Create sentiment labels (positive, negative, neutral) based on rating
def create_sentiment_labels(rating):
    if rating >= 7:
        return 'positive'
    elif rating <= 4:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(create_sentiment_labels)

print("\n--- Preprocessed Text and Sentiment Labels ---")
print(df[['clean_review', 'sentiment']].head())


# In[10]:


# 5. Vectorize Text
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['clean_review'])


# In[11]:


# 6. Prepare Targets
y_class = LabelEncoder().fit_transform(df['sentiment'])
y_reg = df['rating']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)


# In[17]:


# 7. Classification Models
print("Logistic Regression (Classification)")
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train_c, y_train_c)
y_pred_lr = lr_clf.predict(X_test_c)
print(classification_report(y_test_c, y_pred_lr))


# In[18]:


print("Random Forest Classifier ")
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict(X_test_c)
print(classification_report(y_test_c, y_pred_rf))


# In[19]:


# 8. Regression Models
print("Linear Regression")
lr_reg = LinearRegression()
lr_reg.fit(X_train_r, y_train_r)
y_pred_lr_reg = lr_reg.predict(X_test_r)
print("MAE:", mean_absolute_error(y_test_r, y_pred_lr_reg))
print("RMSE:",np.sqrt(mean_squared_error(y_test_r, y_pred_lr_reg)))
print("R2:", r2_score(y_test_r, y_pred_lr_reg))


# In[23]:


from sklearn.ensemble import GradientBoostingRegressor

print("Gradient Regression")
gr_reg = GradientBoostingRegressor()
gr_reg.fit(X_train_r, y_train_r)
y_pred_gr_reg = gr_reg.predict(X_test_r)
print("MAE:", mean_absolute_error(y_test_r, y_pred_gr_reg))
print("RMSE:",np.sqrt(mean_squared_error(y_test_r, y_pred_gr_reg)))
print("R2:", r2_score(y_test_r, y_pred_gr_reg))


# In[26]:


# 9. Top Drug Recommendations Based on Positive Sentiment
top_positive = df[df['sentiment'] == 'positive']
# Changed 'DrugName' to 'drugName'
top_drugs = top_positive.groupby(['condition', 'drugName'])['rating'].mean().reset_index()
top_drugs = top_drugs.sort_values(['condition', 'rating'], ascending=[True, False]).groupby('condition').head(3)

print(" Top Recommended Drugs")
print(top_drugs)

