# %% [markdown]
# # Process data

# %%
import gzip
import json
import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm  

# Process entire dataset if needed

def process_reviews():
    dataset = []
    path = "Electronics.jsonl.gz"

    with gzip.open(path, 'rt', encoding="utf8") as f:
        total_lines = sum(1 for _ in f)

    with gzip.open(path, 'rt', encoding="utf8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing reviews"):
            d = json.loads(line.strip())
            processed_review = {
                'user_id': d['user_id'],
                'rating': int(float(d['rating'])),
                'helpful_vote': int(d['helpful_vote']),
                'timestamp': int(d['timestamp']),
                'asin': d['asin'],
                'text': d['text'],
                'title': d['title'],
                'parent_asin': d['parent_asin'],
                'verified_purchase': d['verified_purchase']
            }
            dataset.append(processed_review)
    
    return dataset

# Process sample of dataset

def process_reviews_sample(sample_size=100000):
   dataset = []
   path = "Electronics.jsonl.gz"
   
   with gzip.open(path, 'rt', encoding="utf8") as f:
       for i, line in tqdm(enumerate(f), total=sample_size, desc="Processing sample"):
           if i >= sample_size:
               break
           d = json.loads(line.strip())
           processed_review = {
               'user_id': d['user_id'],
               'rating': int(float(d['rating'])),
               'helpful_vote': int(d['helpful_vote']),
               'timestamp': int(d['timestamp']),
               'asin': d['asin'],
               'text': d['text'],
               'title': d['title'],
               'parent_asin': d['parent_asin'],
               'verified_purchase': d['verified_purchase']
           }
           dataset.append(processed_review)
   
   return dataset

sample_dataset = process_reviews_sample(100000)

# %% [markdown]
# # Exploratory analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert dataset to pandas DataFrame for easier analysis
df = pd.DataFrame(sample_dataset)

# Basic statistics
print("\nBasic Statistics:")
print(df['rating'].describe())

# Rating distribution
plt.figure(figsize=(10, 6))
df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Average rating over time
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
monthly_ratings = df.groupby(df['date'].dt.to_period('M'))['rating'].mean()
plt.figure(figsize=(12, 6))
monthly_ratings.plot(kind='line')
plt.title('Average Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.show()

# Helpful votes analysis
print("\nHelpful Votes Statistics:")
print(df['helpful_vote'].describe())

# Verified vs unverified purchase ratings
print("\nAverage Rating by Verified Purchase:")
print(df.groupby('verified_purchase')['rating'].mean())

# %%



