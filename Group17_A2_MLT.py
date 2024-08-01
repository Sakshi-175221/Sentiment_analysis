#!/usr/bin/env python
# coding: utf-8

# <p style="font-family: Times; font-size:2.5em;color:darkgoldenrod;"> NLP: Sentiment Analysis </p>

# <p style="font-family: Times; font-size:1.5em;color:coral;"> Sentiment Analysis of Restaurant Reviews </p>
# Sentiment analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral. 

# <p style="font-family: Times; font-size:1.5em;color:chocolate;"> Student Details: <ol>
#     <li><b>A049 Heeta Parmar </b></li> 
#     <li><b>A064 Sakshi Shinde</b></li> 
#     <li><b>A047 Madhumayi Parakala</b></li>  </p>

# <p style="font-family: Times; font-size:1.5em;color:chocolate;"> Data Description:</p>
# Drive Link: https://drive.google.com/drive/folders/1J4YJO6Sz5-Hio_8HeQMRl_XQ2kFXvTQG <ol>
#     The dataset contains two columns 'Review' and 'Liked'.
# Liked variable takes two values 0 and 1.
# The motive of this project is to classify the review into positive or negative. It is a balanced data.

# <p style="font-family: Times; font-size:1.5em;color:chocolate;"> Models Used:<ol></p>
#     <li><b>Stopwords</b></li> library is imported to eliminate words that are so widely used that they carry very little useful information.
#     <li><b>Stemmers</b></li> are the algorithms used to reduce different forms of a word to a base form. Essentially, they do this by removing specific character strings from the end of word tokens.
#   <li><b> CountVectorizer</b ></li>is used to transform the words into vectors on the basis of count.
#   <li><b> Multinomial Naive Bayes</b ></li> is suitable for classification with discrete features (e.g., word counts for text classification)

# <p style="font-family: Times; font-size:1.5em;color:chocolate;"> Learnings: </p>
#     Learnt about how data preprocessing plays a significant role before fitting the model. Eliminating stopwords can improve the accuracy and relevance of NLP tasks by drawing attention to the more important words, or content words.
#     Stemming is a method in text processing that eliminates prefixes and suffixes from words, transforming them into their fundamental or root form.
#     CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is nothing but the count of the word in that particular text sample.

# In[4]:


# Importing essential libraries
import numpy as np
import pandas as pd


# In[5]:


df=pd.read_csv(r"C:\Users\sonam\Downloads\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)


# In[3]:


df.head()


# In[6]:


df['Liked'].value_counts()


# In[5]:


df.shape


# In[6]:


df.columns


# <div class='alert alert-block alert-info'>
#      <b>Data Preprocessing</b>
# </div>
#     

# In[7]:


# Importing essential libraries for performing Natural Language Processing on 'Restaurant_Reviews.tsv' dataset
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[8]:


# Cleaning the reviews
corpus = []
for i in range(0,1000):

  # Cleaning special character from the reviews
  review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=df['Review'][i])

  # Converting the entire review into lower case
  review = review.lower()

  # Tokenizing the review by words
  review_words = review.split()

  # Removing the stop words
  review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

  # Stemming the words
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review_words]

  # Joining the stemmed words
  review = ' '.join(review)

  # Creating a corpus
  corpus.append(review)


# In[9]:


corpus[0:10]


# <div class='alert alert-block alert-info'>
#      <b>Model Building</b>
# </div>

# In[10]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[12]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[13]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[14]:


# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("---- Scores ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))
print("Precision score is: {}".format(round(score2,2)))
print("Recall score is: {}".format(round(score3,2)))


# In[15]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[16]:


cm


# In[17]:


# Plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (10,6))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')


# In[18]:


# Hyperparameter tuning the Naive Bayes Classifier
best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.1,1.1,0.1):
  temp_classifier = MultinomialNB(alpha=i)
  temp_classifier.fit(X_train, y_train)
  temp_y_pred = temp_classifier.predict(X_test)
  score = accuracy_score(y_test, temp_y_pred)
  print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))
  if score>best_accuracy:
    best_accuracy = score
    alpha_val = i
print('--------------------------------------------')
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))


# In[19]:


classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)


# <div class='alert alert-block alert-info'>
#      <b>Prediction</b>
# </div>

# In[20]:


def predict_sentiment(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = cv.transform([final_review]).toarray()
  return classifier.predict(temp)


# In[21]:


# Predicting values
sample_review = 'The food is really good here.'

if predict_sentiment(sample_review):
  print('This is a POSITIVE review.')
else:
  print('This is a NEGATIVE review!')


# In[22]:


# Predicting values
sample_review = 'Food was pretty bad and the service was very slow.'

if predict_sentiment(sample_review):
  print('This is a POSITIVE review.')
else:
  print('This is a NEGATIVE review!')


# In[23]:


# Predicting values
sample_review = 'The food was absolutely wonderful, from preparation to presentation, very pleasing.'

if predict_sentiment(sample_review):
  print('This is a POSITIVE review.')
else:
  print('This is a NEGATIVE review!')


# <p style="font-family: Times; font-size:1.5em;color:chocolate;"> Future Scope: </p>
#     We can try using lemmatization instead of stemming and compare the results.
#     We can try fitting other models like SVM. 

# In[ ]:




