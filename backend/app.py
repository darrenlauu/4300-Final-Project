import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import feedback

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

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this


def sql_search(input_search, countries):
    input_attributes = input_search.split(" ")
    countries_list = countries.split(",")
    input_attributes = list(map(lambda x: x.strip(), input_attributes))

    # like_text_full = "country IN (" + \
    #     ",".join([f"'{c}'" for c in countries_list]) + ")"

    query_sql = f"""SELECT hotel_name, Positive_Review, review_id, country FROM reviews"""
    keys = ["Hotel_Name", "Positive_Review", "review_id", "country"]
    data = mysql_engine.query_selector(query_sql)
    reviews = [dict(zip(keys, i)) for i in data]
    if os.path.exists("tfidf-doc.p"):
        doc_by_vocab = pickle.load(open("tfidf-doc.p", 'rb'))
        index_to_vocab = pickle.load(open("index_to_vocab.p", 'rb'))
        vocab_to_index = pickle.load(open("vocab_to_index.p", 'rb'))
    else:
        tfidf_vec = TfidfVectorizer(max_features=500,
                                    stop_words="english",
                                    max_df=0.1,
                                    min_df=10,
                                    norm='l2')
        doc_by_vocab = tfidf_vec.fit_transform(
            [d['Positive_Review'] for d in reviews]).toarray()

        pickle.dump(doc_by_vocab, open("doc_by_vocab.p", 'wb'))
        pickle.dump(tfidf_vec, open("tfidf_vec.p", 'wb'))

        index_to_vocab = {i: v for i, v in enumerate(
            tfidf_vec.get_feature_names_out())}
        pickle.dump(index_to_vocab, open("index_to_vocab.p", 'wb'))
        vocab_to_index = {v: i for i, v in enumerate(
            tfidf_vec.get_feature_names_out())}
        pickle.dump(vocab_to_index, open("vocab_to_index.p", 'wb'))
    print(doc_by_vocab.shape)
    print(type(doc_by_vocab))
    print(doc_by_vocab[0])
    query_vec = np.zeros(len(index_to_vocab))
    for tkn in input_attributes:
        if tkn in vocab_to_index:
            ind = vocab_to_index[tkn]
            query_vec[ind] += 1

    cos_score = np.zeros(len(reviews))

    query_vec = feedback.rocchio(query_vec, doc_by_vocab, dict())

    for i in range(len(reviews)):
        q_norm = np.linalg.norm(query_vec)
        d_norm = np.linalg.norm(doc_by_vocab[i])
        if q_norm == 0 or d_norm == 0:
            score = 0
        else:
            score = np.dot(doc_by_vocab[i], query_vec)/(q_norm*d_norm)
        cos_score[i] = score

    # ans = np.argmax(cos_score)
    # print(processed[ans][1], processed[ans][2])
    top10 = list(np.flip(np.argsort(cos_score))[:10])
    ret_keys = ["Hotel_Name", "Positive_Review"]
    ans = [(reviews[i]['Hotel_Name'], reviews[i]["Positive_Review"])
           for i in top10]
    return json.dumps([dict(zip(ret_keys, i)) for i in ans])


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    countries = request.args.get("countries")
    return sql_search(text, countries)


if not mysql_engine.IS_DOCKER:
    app.run(debug=True)
