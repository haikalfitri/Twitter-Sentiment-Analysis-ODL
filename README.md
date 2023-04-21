# Twitter Sentiment Analysis of Open Distance Learning
This project aims to analyze the sentiment of tweets related to Open Distance Learning (ODL) using natural language processing techniques. The sentiment analysis was conducted on a large dataset of tweets related to ODL, using machine learning algorithms to classify the tweets as positive, negative, or neutral.

## 1. Introduction

## Problem Statement 
- Manually analyzing large volumes of tweets related to ODL is time-consuming and impractical.

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

## Data Label
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

Next, filtered the dataset to include only the tweets that are in English language and save the file for the processing.
```python
#filter tweets in english only
eng_only = data_merged.loc[data_merged['language']=="en"]

#save file
eng_only.to_csv('Labelled Dataset.csv', header=True)
```
