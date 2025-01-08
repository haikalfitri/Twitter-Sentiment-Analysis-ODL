# Twitter Sentiment Analysis of Open Distance Learning
This project aims to analyze the sentiment of tweets related to Open Distance Learning (ODL) using natural language processing techniques. The sentiment analysis was conducted on a large dataset of tweets related to ODL, using machine learning algorithms to classify the tweets as positive, negative, or neutral.

To view the interactive dashboard, download [here]([https://muhdhaikalfiri.shinyapps.io/MonsterMovieAnalysisandPredictionApp/](https://github.com/haikalfitri/Twitter-Sentiment-Analysis-ODL/blob/main/Sentiment%20Analysis%20-%20ODL.pbix)).

<p align="center">
  <img src="[https://github.com/haikalfitri/Monster-Movie-Analysis-and-Prediction-App-using-R/blob/main/asset/ss%20monster%20movie%20analysis.png](https://github.com/haikalfitri/Twitter-Sentiment-Analysis-ODL/blob/main/Asset/2nd%20project%20sentiment%20analysis.png)" alt="First Page" width="600" />
</p>
<p align="center"><strong>Preview</strong></p>

## Problem Statement 
- Manually analyzing large volumes of tweets related to ODL is time-consuming and impractical.
- Assess public sentiment towards ODL to understand perceptions.
- Determine the factors influencing public perceptions of ODL, i

## Project Scope
- Tools: Jupyter Notebook, RapidMiner Studio, Microsoft Power Bi, TWINT
- Dataset: Raw = around 100,000 tweets, Preprocessed: around 60,000 tweets
- Tweets Range: 2-3 years span from 2022

## Dataset Collection
To collect the dataset for this project, I used the TWINT library to scrape tweets related to my search query using the following code:
```python
#import the twint library
import twint

#set the configuration options
c = twint.Config()
c.Search = "distraction online learning"
c.Limit = 1500
c.Count= True
c.Lang = "en"
c.Until = '2022-11-22 10:00:00'
c.Since = '2022-01-01 10:00:00' 
c.Store_csv = True
c.Output = "scrape1.csv"

#execute the search
twint.run.Search(c)
```
This code sets the configuration options for the TWINT scraper, including the search term, language, and date range. The search is then executed using these configuration options, and the resulting tweets are saved in CSV files. This project involved scraping Twitter for tweets related to seven different keywords/topics - 'adaptation', 'distraction', 'equipment', 'internet', 'mental health', 'motivation', and 'time management'. I collected a total of around 100,000 tweets for this project.

## Topic Label
After scraping tweets, I merged the datasets into a single dataset using Pandas' 'concat' function. First step is the datasets files are assign to dataframe.
```python
#import necessary libraries
import pandas as pd
import numpy as np

# assign the dataframe and read in data for each topic from your save file
df_adaptation = pd.read_csv('FYP LATEST/2. Merged Dataset/Adaptation merged.csv')
df_adaptation.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_adaptation.shape

df_distraction = pd.read_csv('FYP LATEST/2. Merged Dataset/Distraction merged.csv')
df_distraction.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_distraction.shape

df_equipment = pd.read_csv('FYP LATEST/2. Merged Dataset/Equipment merged.csv')
df_equipment.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_equipment.shape

df_internet = pd.read_csv('FYP LATEST/2. Merged Dataset/Internet access merged.csv')
df_internet.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_internet.shape

df_mentalhealth = pd.read_csv('FYP LATEST/2. Merged Dataset/Mental health merged.csv')
df_mentalhealth.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_mentalhealth.shape

df_motivation = pd.read_csv('FYP LATEST/2. Merged Dataset/Motivation merged.csv')
df_motivation.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_motivation.shape

df_timemanage = pd.read_csv('FYP LATEST/2. Merged Dataset/Time management merged.csv')
df_timemanage.drop_duplicates(subset = 'tweet', keep = 'first', inplace = True)
df_timemanage.shape
```
Then remove unwanted column for each dataframe. For this project, I only keep 3 important column which is
- tweet
- date 
- language
```python
df1_adaptation= df_adaptation[['tweet', 'date','language']]
df1_distraction= df_distraction[['tweet', 'date','language']]
df1_equipment= df_equipment[['tweet', 'date','language']]
df1_internet= df_internet[['tweet', 'date','language']]
df1_mentalhealth= df_mentalhealth[['tweet', 'date','language']]
df1_motivation= df_motivation[['tweet', 'date','language']]
df1_timemanage= df_timemanage[['tweet', 'date','language']]
```

To label the topic of each tweet, I used Pandas' assign() function to create a new column for each dataset and assign the corresponding topic to each row. For example, to label the tweets  in the datasets for each topics, I used the following code:
```python
#assign() function
df_adaptation= df_adaptation.assign(Topic = 'Adaptation')
df_distraction= df_distraction.assign(Topic = 'Distraction')
df_equipment= df_equipment.assign(Topic = 'Equipments')
df_internet= df_internet.assign(Topic = 'Internet Access')
df_mentalhealth= df_mentalhealth.assign(Topic = 'Mental Health')
df_motivation= df_motivation.assign(Topic = 'Motivation')
df_timemanage= df_timemanage.assign(Topic = 'Time management')
```

Next, merged the dataframe using Pandas() concat function. The code below concatenates the data frames for each topic into a single data frame, which makes it easier to preprocess the data and perform sentiment analysis.
```python
#assign() function
frames = [df_adaptation,df_distraction,df_equipment,df_internet,df_internet,df_mentalhealth,df_motivation,
          df_timemanage ] 
data_merged = pd.concat(frames, ignore_index=True);
```

Next, filtered the dataset to include only the tweets that are in English language and save the file for the data pre-processing.
```python
#filter tweets in english only
eng_only = data_merged.loc[data_merged['language']=="en"]

#save file
eng_only.to_csv('Labelled Dataset.csv', header=True)
```

## Data Pre-Processing

### Import Libraries
```python
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import nltk
import contractions
#import ssl
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize 
#nltk.download('punkt')
#nltk.download('sentiwordnet')
#nltk.download('stopwords') 
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')

pd.set_option("display.max_colwidth", -1)
pd.options.mode.chained_assignment = None  # default='warn'
```

### a) Remove Duplicate
Remove the duplicate to prevent skewed analysis results.
```python
#removes duplicate rows based on the preprocess_tweet column
df.drop_duplicates(subset = 'preprocess_tweet', keep = 'first', inplace = True)
```
### b) Expand Contractions
Replacing shortened forms of words (like "can't" for "cannot" or "I've" for "I have") with their full forms in text data.
```python
#expand the contraction of text in tweets
df['contraction'] = df['preprocess_tweet'].apply(lambda x: contractions.fix(x))
df
```
### c) Lower Tweets
Lowercasing the text to ensuring uniformity and simplify text comparisons
```python
#to transform all the text in the tweets to the lower case
df['lower_tweet'] = df['contraction'].str.lower()
df
```

### d) Remove Numbers
Removing numbers as analysis focus on textual data rather than numerical value
```python
#to remove the number 
def remove_num(text):
    text = re.sub('[0-9]', '', text)
    text = text.strip()
    return text

df['remove_num']= df['lower_tweet'].apply(lambda x:remove_num(x))
df
```
#### e) Remove URLs
To ensure URLs does not interfere with the analysis
```python
#to remove urls
def remove_URL(text):
    URL = re.compile(r'https?://\S+|www\.\S+')
    return URL.sub(r'', text)


df['remove_urls'] = df['remove_num'].apply(remove_URL)
df
```

### f) Remove Hashtags
To avoid tweets biasing based on topics or trends indicated by hashtags
```python
#to remove hashtags 
def remove_Hashtag(text):
    Hashtag = re.compile(r'#\S+|www\.\S+')
    return Hashtag.sub(r'', text)
df['remove_hashtags'] = df['remove_urls'].apply(remove_Hashtag)
df
```
### g) Remove Mentions
To excludes user handles from the text
```python
#to remove mentions from the tweets
def remove_Mentions(text):
    Mentions = re.compile(r'@[A-Za-z0-9._/]+')
    return Mentions.sub(r'', text)

df['remove_mentions'] = df['remove_hashtags'].apply(remove_Mentions)
df
```
### h) Remove Emoji
Eliminate graphical symbols or graphics from the texts
```python
#to remove emojis from the tweets
filter_char = lambda c: ord(c) < 256
df['remove_emoji'] = df['remove_mentions'].apply(lambda s: ''.join(filter(filter_char, s)))
df
```

### i) Remove Punctuations
Standardizing text representation and improving the accuracy
```python
# to remove the punctuations
df['remove_punctuation'] = df['remove_emoji'].str.replace('[^\w\s]', '')
df
```

### j) Remove Custom Stopword
To exclude specific words or phrases that are deemed irrelevant to the analysis
```python
stopwords_nltk = stopwords.words('english')

stopwords_nltk.append('even')
stopwords_nltk.append('ever')
stopwords_nltk.append('etc')
stopwords_nltk.append('still')
stopwords_nltk.append('say')
stopwords_nltk.append('im')
stopwords_nltk.append('tell')
stopwords_nltk.append('us')
stopwords_nltk.append('thats')
not_stopwords_nltk = {'no','should', 'not','o','over','nor','very', 'did',"should've", "you've", 'does' 'not','t','can','against', 'do','don',"don't",'ain','aren',"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", }
real_stopwords_nltk = set([word for word in stopwords_nltk if word not in not_stopwords_nltk])

def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in real_stopwords_nltk])

df["remove_stopwords"] = df["remove_punctuation"].apply(stopwords)
df
```

### k) Tokenization
Preparing text data for analysis
```python
#to tokenize

def tokenization(text):
    text = re.split('\W+', text)
    return text

df['tokenization'] = df['remove_stopwords'].apply(lambda x: tokenization(x.lower()))
df
```
### l) Lemmatization
To convert the words to the based form
```python
#lemmatization

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} 

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text)
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df['lemmatization'] = df['tokenization'].apply(lemmatize_words)
df.tail()
```

## Sentiment Labelling

The tweets that are collected will be labelled into three classes: positive (+1), neutral (0), and negative (-1). The process will go through RapidMiner to generate sentiments score and label the polarity of the text. The parameter operator provides access to switch between the desired models used for the project: Vader and SentiWordNet.

![image](https://github.com/haikalfitri/Twitter-Sentiment-Analysis-ODL/assets/105991397/8a1b36ea-9460-4823-ae08-06fa577f8bfe)

## Modeling
Three machine learning techniques were employed in this project: Naive Bayes (NB), Support Vector Machine (SVM), and Random Forest (RF). The labeled dataset was divided into training and testing datasets using split ratios of 80:20. The training dataset was used to train each machine learning model, while the testing dataset was used to evaluate their classification performance. The models were evaluated based on their ability to classify positive and negative sentiments. The Scikit-learn library in Python was utilized for modeling and evaluation.

Splits the dataset into training and testing sets using a 80:20 ratio with stratification. Utilizes TF-IDF vectorization to convert text into numerical features for machine learning models, ensuring compatibility with Scikit-learn classifiers.
```python
# Splitting data into X (features) and y (target)
X = new_df['preprocess_tweet']
y = new_df['Label']

# Importing necessary libraries
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45, stratify=y)

# Initializing and fitting the TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.astype("U").str.lower())
X_test_vec = vectorizer.transform(X_test.astype("U").str.lower())
```
This project employs three machine learning techniques to perform sentiment analysis on preprocessed text data:

Multinomial Naive Bayes (NB): Trained and evaluated using the MultinomialNB classifier.
Support Vector Machine (SVM): Implemented with the LinearSVC classifier.
Random Forest (RF): Applied using the RandomForestClassifier.
Each model is trained on TF-IDF transformed text data, split into training and testing sets with a ratio of 80:20. Model performance is evaluated using accuracy, confusion matrix, and classification report metrics. The Scikit-learn library in Python is used for implementation and evaluation.

### a) NB
```python
    %%time
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()

    mnb.fit(X_train_vec, y_train)
    #nb.fit(X_train_vec, y_train)

    # make class predictions for X_test_vec
    y_pred_class_mnb = mnb.predict(X_test_vec)

    from sklearn import metrics

    # calculate accuracy of class predictions
    print('accuracy: %.4f' % metrics.accuracy_score(y_test, y_pred_class_mnb))

    # confusion matrix
    print('confusion matrix:')
    print(metrics.confusion_matrix(y_test, y_pred_class_mnb))

    # classification report
    print('classification report:')
    print(metrics.classification_report(y_test, y_pred_class_mnb,digits = 4))
```

### b) SVM
```python
%%time
from sklearn.svm import LinearSVC
lsvc = LinearSVC(random_state=45, tol=1e-5)

#clf_sv = GridSearchCV(svcl, params)
lsvc.fit(X_train_vec, y_train)

# make class predictions for X_test_vec
y_pred_class_lsvc = lsvc.predict(X_test_vec)

from sklearn import metrics

# calculate accuracy of class predictions
print('accuracy: %.3f' % metrics.accuracy_score(y_test, y_pred_class_lsvc))

# confusion matrix
print('confusion matrix:')
print(metrics.confusion_matrix(y_test, y_pred_class_lsvc))

# classification report
print('classification report:')
print(metrics.classification_report(y_test, y_pred_class_lsvc,digits = 4))
```

### c) Random Forest
```python
%%time
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=45)
rf.fit(X_train_vec, y_train)

# make class predictions for X_test_vec
y_pred_class_rf = rf.predict(X_test_vec)

from sklearn import metrics

# calculate accuracy of class predictions
print('accuracy: %.3f' % metrics.accuracy_score(y_test, y_pred_class_rf))

# confusion matrix
print('confusion matrix:')
print(metrics.confusion_matrix(y_test, y_pred_class_rf))

# classification report
print('classification report:')
print(metrics.classification_report(y_test, y_pred_class_rf,digits = 4))
```
Repeat the modeling process using different sentiment lexicons (e.g., SentiWordNet or VADER) and different train-test split ratios (80:20 and 60:40).











