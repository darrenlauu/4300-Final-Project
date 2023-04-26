import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import feedback
import operator

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = os.environ.get("MYSQL_ROOT_PASSWORD")
MYSQL_PORT = 3306
MYSQL_DATABASE = "kardashiandb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# mysql_engine.query_executor(f"USE {MYSQL_DATABASE};")

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

print("TVIBESLOG: Querying reviews for cosine sim initialization")

def get_reviews():
    query_sql = f"""SELECT
    H.hotel_name, Positive_Review, review_id, H.country, H.hotel_id, H.average_score, H.Topic_1, H.Topic_2, H.Topic_3, H.Topic_4, H.Topic_5
    FROM reviews R
    join hotels H on R.hotel_id = H.hotel_id
    """
    keys = ["Hotel_Name", "Positive_Review",
            "review_id", "Country", "Hotel_ID", "Average_Score", 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
    data = mysql_engine.query_selector(query_sql)
    reviews = [dict(zip(keys, i)) for i in data]
    return reviews

# if os.path.exists("vectorizer.p"):
#     print("TVIBESLOG: Loading TFIDF from pickle")
#     vectorizer = pickle.load(open("vectorizer.p", "rb"))
#     tfIdfMatrix = pickle.load(open("tfIdfMatrix.p", "rb"))
# else:
print("TVIBESLOG: Pickling TFIDF from SQL")
reviews = get_reviews()
reviews_list = [d['Positive_Review'] for d in reviews]
print("TVIBESLOG: Made the list of reviews")

vectorizer = TfidfVectorizer(max_features=500,
                            stop_words="english",
                            max_df=0.1,
                            min_df=10,
                            norm='l2')
tfIdfMatrix = vectorizer.fit_transform(reviews_list)

pickle.dump(vectorizer, open("vectorizer.p", "wb"))
pickle.dump(tfIdfMatrix, open("tfIdfMatrix.p", "wb"))

    
print("TVIBESLOG: Made the TF-IDF matrix")

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this


def agg_reviews(reviews):
    dic = {}
    id_to_hot = {}
    for review in reviews:
        hotel_name, hotel_id, rev = review["Hotel_Name"], review["Hotel_ID"] - \
            1, review['Positive_Review']
        if hotel_id not in dic:
            dic[hotel_id] = rev
        else:
            dic[hotel_id] += rev
        if hotel_id not in id_to_hot:
            id_to_hot[hotel_id] = hotel_name
    return dic, id_to_hot


def find_section(text, attr):
    lst = []
    for i in attr:
        val = text.find(i)
        if val >= 0:
            lst.append(i)
    return min(lst), max(lst)

def sql_search(reviews, input_search, countries):
    countries_list = countries.split(",")
    countries_set = set(countries_list)

    query = vectorizer.transform([input_search])
    similarities = cosine_similarity(query, tfIdfMatrix)
    print("TVIBESLOG: Computed the cosine similarity")

    index_to_similarities = {}
    for i in range(len(similarities[0])):
        index_to_similarities[i] = similarities[0][i]
    print("TVIBESLOG: Made the index to similarity dictionary")

    sorted_indices = dict(sorted(index_to_similarities.items(), key=operator.itemgetter(1), reverse=True)[:1000]).keys()
    keys = ["Hotel_Name", "Positive_Review",
            "review_id", "Country", "Hotel_ID", "Average_Score", 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
    sql_query = f"""
    SELECT
    H.hotel_name, Positive_Review, review_id, H.country, H.hotel_id, H.average_score, H.Topic_1, H.Topic_2, H.Topic_3, H.Topic_4, H.Topic_5
    FROM reviews R
    join hotels H on R.hotel_id = H.hotel_id
    where review_id in ({",".join([str(i+1) for i in sorted_indices])})"""
    data = mysql_engine.query_selector(sql_query)
    result = [dict(zip(keys, i)) for i in data]
    print("TVIBESLOG: Sorted the indices and got the top 10 reviews")
    return json.dumps(list(filter(lambda x: True, filter(lambda x: x["Country"] in countries_set, result)))[:10])


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")

@ app.route("/<int:hotel_id>/")
def hotel(hotel_id):
    hotel_keys = ["Hotel_Name", "Country", "Hotel_ID", "Average_Score", 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
    hotel_sql = f"""SELECT {",".join(hotel_keys)} FROM hotels WHERE hotel_id = {hotel_id}"""
    hotel = mysql_engine.query_selector(hotel_sql)
    name, country, hid, score, t1, t2, t3, t4, t5 = [dict(zip(hotel_keys, i)) for i in hotel][0].values()

    reviews_sql = f"""SELECT Positive_Review FROM reviews WHERE hotel_id = {hotel_id}"""
    review_keys = ["Positive_Review"]
    data = mysql_engine.query_selector(reviews_sql)
    reviews = list(map(lambda x: x["Positive_Review"], [dict(zip(review_keys, i)) for i in data]))
    return render_template('hotel.html', name=name, country=country, hid=hid, score=float(score), tags=[t1, t2, t3, t4, t5], reviews=reviews)


@ app.route("/reviews")
def reviews_search():
    text = request.args.get("title")
    countries = request.args.get("countries")
    return sql_search(None, text, countries)

@ app.route("/upvote/<int:hotel_id>/", methods=["POST"])
def upvote(hotel_id):
    plus = request.args.get("upvote") == "true" # anything else is false
    if plus:
        upvote_op = "+"
    else:
        upvote_op = "-"
    mysql_engine.query_executor(f"UPDATE reviews SET upvote = upvote {upvote_op} 1 WHERE hotel_id={hotel_id}")
    return json.dumps({"success":True})


if not mysql_engine.IS_DOCKER:
    app.run(debug=True)
