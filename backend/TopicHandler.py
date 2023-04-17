import json
import os
import string
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

translator = str.maketrans('', '', string.punctuation)

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = os.environ.get("MYSQL_ROOT_PASSWORD")
MYSQL_PORT = 3306
MYSQL_DATABASE = "kardashiandb"

mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

mysql_engine.query_executor(f"USE {MYSQL_DATABASE};")

mysql_engine.query_executor("""
CREATE TABLE IF NOT EXISTS `reviews`(
`Hotel_Address` TEXT, `Average_Score` TEXT,
 `Hotel_Name` TEXT, `Negative_Review` TEXT, `Review_Total_Negative_Word_Counts` TEXT,
 `Total_Number_of_Reviews` TEXT, `Positive_Review` TEXT, `Review_Total_Positive_Word_Counts` TEXT, `Reviewer_Score` TEXT);
""")

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file

# This is a check to see if the DB is empty, if it is, then we load all the reviews
review_count = mysql_engine.query_selector("select count(*) from reviews")
if sum([c for c in review_count][0]) == 0:
    print("TVIBESLOG: Loading Hotel Reviews entries for the first time")
    mysql_engine.load_file_into_db()

else:
    print("TVIBESLOG: Hotel Reviews entries already exist")

mysql_engine.query_executor(f"USE {MYSQL_DATABASE};")

app = Flask(__name__)
CORS(app)


def get_10000_reviews():
  query_sql = f"""SELECT hotel_name, Positive_Review FROM reviews LIMIT 10000;"""
  data = mysql_engine.query_selector(query_sql)
  return data.mappings().all()

def build_review_matrix(reviews):
  """
  takes a list of reviews as input and creates a matrix representation of the reviews 
  using the Term Frequency-Inverse Document Frequency (TF-IDF) technique. It uses the `TfidfVectorizer` 
  class from scikit-learn library to vectorize the reviews and generate a sparse matrix of term 
  frequencies. The function returns the review matrix and a list of the feature names extracted from 
  the reviews.
  """
  vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, max_df=0.05, smooth_idf=True)
  review_matrix = vectorizer.fit_transform(reviews)
  return review_matrix, vectorizer.get_feature_names()


def extract_topics_svd(reviews):
  svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)

  hotel_reviews = {}
  for review in reviews:
    hotel_name = review['hotel_name']
    if hotel_name not in hotel_reviews:
      hotel_reviews[hotel_name] = []
    hotel_reviews[hotel_name].append(review['Positive_Review'])

  topics_data = []
  for hotel_name, reviews in hotel_reviews.items():
    review_matrix, feature_names = build_review_matrix(reviews)
    svd.fit_transform(review_matrix)
    topics = dict()

    for i in range(min(len(reviews), 10)):
      for j in svd.components_[i].argsort()[::-1]:
        if feature_names[j] not in topics.keys():
          topics[feature_names[j]] = svd.components_[i][j]
        else:
          topics[feature_names[j]] += svd.components_[i][j]
    
    topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
    topics_data.append({'Hotel_Name': hotel_name, 'Topics': topics})

  for i in range(len(topics_data)):
    print(topics_data[i])
  return topics_data




def extract_topics_lda(reviews):
  review_matrix, feature_names = build_review_matrix(reviews)
  lda = LatentDirichletAllocation(n_components=10, random_state=42)
  lda.fit(review_matrix)
  topics_data = []
  for i, review in enumerate(reviews):
    hotel_name = review['hotel_name']
    review_text = review['Positive_Review']
    topic_weights = np.array(lda.components_[i])
    topics = [(feature_names[j], topic_weights[j]) for j in topic_weights.argsort()[:-11:-1]]
    topics_data.append({'Hotel_Name': hotel_name, 'Positive_Review': review_text, 'Topics': topics})
  print(topics_data[0])
  return topics_data

extract_topics_svd(get_10000_reviews())
# extract_topics_lda(get_10_reviews())
