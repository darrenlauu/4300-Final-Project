import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np
import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
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

REVIEWS_PER_PAGE = 10
ALL_TAGS = {'breakfast', 'lovely', 'atmosphere', 'fantastic', 'quiet', 'rooms', 'facilities', 'comfy', 'nice', 'modern', 'decor', 'bed', 'beds', 'service', 'perfect', 'really', 'clean', 'attention', 'professional', 'amazing', 'extra', 'buffet', 'accommodating', 'helpful', 'excellent', 'comfort', 'good', 'stylish', 'decoration', 'beautiful', 'comfortable', 'brilliant', 'positive', 'super', 'bathroom', 'best', 'exceptional', 'design', 'near', 'fabulous', 'price', 'easy', 'champagne', 'loved', 'polite', 'victoria', 'upgraded', 'attentive', 'small', 'spacious', 'cleanliness', 'central', 'food', 'vibe', 'feel', 'ideal', 'boutique', 'view', 'kitchenette', 'size', 'location', 'wonderful', 'friendliness', 'royal', 'style', 'area', 'place', 'great', 'reception', 'du', 'croissants', 'new', 'shower', 'underground', 'executive', 'quirky', 'club', 'thing', 'extremely', 'close', 'property', 'kind', 'large', 'friendly', 'hammam', 'superb', 'rooftop', 'charming', 'definitely', 'personal', 'character', 'pool', 'balcony', 'absolutely', 'money', 'exceptionally', 'hospitality', 'spa', 'cozy', 'brand', 'welcoming', 'lobby', 'cookie', 'bar', 'center', 'convenient', 'outstanding', 'restaurant', 'unique', 'overall', 'roof', 'open', 'located', 'ok', 'just', 'delicious', 'big', 'gym', 'room', 'hotel', 'italy', 'courtyard', 'parking', 'shopping', 'value', 'desk', 'quality', 'pillows', 'suite', 'metro', 'connections', 'paris', 'delightful', 'totally', 'old', 'tea', 'help', 'space', 'pleasant', 'employee', 'nearby', 'lock', 'quietness', 'concierge', 'stuff', 'cocktails', 'business', 'luxury', 'baguette', 'canal', 'afternoon', 'helpfull', 'city', 'home', 'treat', 'stay', 'village', 'wifi', 'position', 'gem', 'street', 'real', 'experience', 'special', 'art', 'besi', 'station', 'seine', 'beautifully', 'interior', 'enjoyed', 'gard', 'windows', 'customer', 'mr', 'like', 'free', 'walk', 'couldn', 'restaurants', 'tower', 'swimming', 'cool', 'booking', 'neighborhood', 'republique', 'brasserie', 'newly', 'nicely', 'quite', 'family', 'liked', 'helpfulness', 'public', '10', 'little', 'relaxing', 'night', 'apartment', 'rambla', 'winery', 'kitchen', 'ambience', 'laundry', 'walking', 'cake', 'duomo', 'patrick', 'upgrade', 'awesome', 'distance', 'cosy', 'access', 'familiar', 'st', 'reasonably', 'arra', 'efficient', 'terrace', 'high', 'building', 'trouble', 'maintained', 'decorated', 'thank', 'love', 'class', 'vosges', 'fun', 'calm', 'marais', 'hall', 'shops', 'time', 'theme', 'alcohol', 'safe', 'world', 'cleaning', 'coffee', 'lines', 'renovated', 'complementary', 'smell', 'espanya', 'garden', 'staffs', 'people', 'furniture', 'incredibly', 'tapas', 'views', 'westbahnhof', 'park', 'highly', 'fab', 'vienna', 'concept', 'aswell', 'attractions', 'happy', 'breakfasts', 'centrally', 'fine', 'received', 'lights', 'lounge', 'cooked', 'radio', 'services', 'glass', 'london', 'preserved', 'especially', 'soho', 'cadet', 'court', 'arrival', 'spot', 'tube', 'discreet', 'albert', 'elegant', 'floor', 'recommend', 'star', 'books', 'welcome', 'bit', 'barbican', 'bedding', 'tasty', 'check', 'amenities', 'bright', 'trip', 'lots', 'continental', 'pleasure', 'dinner', 'warm', 'transport', 'priced', 'centre', 'smart', 'tidy', 'look', 'europe', 'given', 'transportation', 'eurostar', 'pleasantries', 'choice', 'day', 'ticket', 'amsterdam', 'selection', 'train', 'neri', 'bars', 'music', 'milan', 'caring', 'armani', 'lot'}

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

def get_reviews(idx=None):
    query_sql = f"""SELECT
    H.hotel_name, Positive_Review, review_id, H.country, H.hotel_id, H.average_score, H.Topic_1, H.Topic_2, H.Topic_3, H.Topic_4, H.Topic_5
    FROM reviews R
    join hotels H on R.hotel_id = H.hotel_id
    """
    if idx is not None:
        query_sql += f"WHERE H.hotel_id = {idx}"
    keys = ["Hotel_Name", "Positive_Review",
            "review_id", "Country", "Hotel_ID", "Average_Score", 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
    data = mysql_engine.query_selector(query_sql)
    reviews = [dict(zip(keys, i)) for i in data]
    return reviews

def agg_reviews(reviews):
    num_hotels = max(set([review["Hotel_ID"] for review in reviews]))
    result = ["" for _ in range(num_hotels)]
    id_to_hotel = dict()
    for review in reviews:
        hotel_name, hotel_id, rev = review["Hotel_Name"], review["Hotel_ID"], review['Positive_Review']
        result[hotel_id - 1] += rev
        if hotel_id not in id_to_hotel:
            id_to_hotel[hotel_id] = hotel_name
    return result, id_to_hotel

if os.path.exists("vectorizer.p"):
    print("TVIBESLOG: Loading TFIDF from pickle")
    vectorizer = pickle.load(open("vectorizer.p", "rb"))
    tfIdfMatrix = pickle.load(open("tfIdfMatrix.p", "rb"))
else:
    print("TVIBESLOG: Pickling TFIDF from SQL")
    reviews = get_reviews()
    reviews_agg, id_to_hotel = agg_reviews(reviews)
    # reviews_list = [d['Positive_Review'] for d in reviews_agg]
    print("TVIBESLOG: Made the list of reviews")

    vectorizer = TfidfVectorizer(max_features=5000,
                                stop_words="english",
                                max_df=0.5,
                                min_df=10,
                                norm='l2')
    tfIdfMatrix = vectorizer.fit_transform(reviews_agg)

    pickle.dump(vectorizer, open("vectorizer.p", "wb"))
    pickle.dump(tfIdfMatrix, open("tfIdfMatrix.p", "wb"))

    
print("TVIBESLOG: Made the TF-IDF matrix")

app = Flask(__name__)
CORS(app)


def find_section(text, attr):
    lst = []
    for i in attr:
        val = text.find(i)
        if val >= 0:
            lst.append(i)
    return min(lst), max(lst)

def sql_search(input_search, countries, tags):
    if countries == "":
        countries_set = {"Italy", "Netherlands", "Spain", "United Kingdom", "France"}
    else:
        countries_list = countries.split(",")
        countries_set = set(countries_list)

    if countries == "":
        tags_set = ALL_TAGS
    else:
        tags_list = tags.split(",")
        tags_set = set(tags_list)
        

    query = vectorizer.transform([input_search])
    similarities = cosine_similarity(query, tfIdfMatrix)
    print("TVIBESLOG: Computed the cosine similarity")

    index_to_similarities = {}
    for i in range(len(similarities[0])):
        index_to_similarities[i] = similarities[0][i]
    print("TVIBESLOG: Made the index to similarity dictionary")
    all_indices = sorted(index_to_similarities.items(), key=operator.itemgetter(1), reverse=True)
    sorted_indices = dict(sorted(index_to_similarities.items(), key=operator.itemgetter(1), reverse=True)[:50]).keys()
    keys = ["Hotel_Name", "Country", "Hotel_ID", "Average_Score", 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
    sql_query = f"""
    SELECT
    H.hotel_name, H.country, H.hotel_id, H.average_score, H.Topic_1, H.Topic_2, H.Topic_3, H.Topic_4, H.Topic_5
    FROM hotels H
    where hotel_id in ({",".join([str(i+1) for i in sorted_indices])})"""
    data = mysql_engine.query_selector(sql_query)
    result = [dict(zip(keys, i)) for i in data]
    hotel_id_to_index = dict()
    for index, hotel in enumerate(result):
        hotel_id_to_index[hotel["Hotel_ID"]] = index

    sorted_result = []
    for hotel_id in sorted_indices:
        idx = hotel_id_to_index[hotel_id+1]
        sorted_result.append(result[idx])
    
    filtered_by_tags = list(filter(lambda x: True, filter(lambda x: x["Topic_1"] in tags_set or x["Topic_2"] in tags_set or x["Topic_3"] in tags_set or x["Topic_4"] in tags_set or x["Topic_5"] in tags_set, sorted_result)))
    filtered_by_countries = list(filter(lambda x: x["Country"] in countries_set, filtered_by_tags))


    print("TVIBESLOG: Sorted the indices and got the top 50 hotels")
    return json.dumps(filtered_by_countries[:25])


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")

@ app.route("/<int:hotel_id>/")
def hotel(hotel_id):
    text = request.args.get("query")
    page_number = request.args.get("page_number", 1)

    hotel_keys = ["Hotel_Name", "Country", "Hotel_ID", "Average_Score", 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
    hotel_sql = f"""SELECT {",".join(hotel_keys)} FROM hotels WHERE hotel_id = {hotel_id}"""
    hotel = mysql_engine.query_selector(hotel_sql)
    name, country, hid, score, t1, t2, t3, t4, t5 = [dict(zip(hotel_keys, i)) for i in hotel][0].values()

    reviews_sql = f"""SELECT Positive_Review, Reviewer_Score FROM reviews WHERE hotel_id = {hotel_id}"""
    data = mysql_engine.query_selector(reviews_sql)
    reviews_and_scores = [(i[0], i[1]) for i in data]
    reviews = list(map(lambda x: x[0], reviews_and_scores))
    scores = list(map(lambda x: float(x[1]), reviews_and_scores))
    words_to_highlight = []
    num_reviews = len(reviews)
    
    if text is not None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.5, min_df=10, norm='l2')
        
        tfIdfMatrix = vectorizer.fit_transform(reviews)
        print("TVIBESLOG: hotel: Made the TF-IDF matrix")

        terms = vectorizer.get_feature_names()
        co_occurrence = np.dot(tfIdfMatrix.T, tfIdfMatrix).toarray()
        print("TVIBESLOG: hotel: Made the co-occurrence matrix")

        query_terms = text.split()
        relevant_terms = [term for term in query_terms if term in terms]
        expanded_query = query_terms

        for term in relevant_terms:
            idx = terms.index(term)
            associated_terms = co_occurrence[idx].argsort()
            associated_terms = associated_terms[::-1][:5]
            for i in associated_terms:
                expanded_query.append(terms[i])
                
        words_to_highlight = list(set(expanded_query))
        print("TVIBESLOG: hotel: Here are the words to highlight:", words_to_highlight)

        # compute similarity with ORIGINAL query, not the expanded one
        query = vectorizer.transform([text])

        similarities = cosine_similarity(query, tfIdfMatrix)
        print("TVIBESLOG: hotel: Computed the cosine similarity")

        index_to_similarities = {}
        for i in range(len(similarities[0])):
            index_to_similarities[i] = similarities[0][i]
        print("TVIBESLOG: hotel: Made the index to similarity dictionary")

        sorted_indices = dict(sorted(index_to_similarities.items(), key=operator.itemgetter(1), reverse=True)[:100]).keys()
        reviews = list(map(lambda idx: reviews[idx], sorted_indices))

    page_number = int(page_number)
    start = (page_number - 1) * REVIEWS_PER_PAGE
    reviews = reviews[start:start + REVIEWS_PER_PAGE]

    print("length of reviews: " + str(num_reviews))
    print(str(REVIEWS_PER_PAGE))
    print(str(math.ceil(num_reviews / REVIEWS_PER_PAGE)))
    
    return render_template(
        'hotel.html',
        name=name, 
        country=country, 
        hid=hid, 
        score=float(score), 
        tags=[t1, t2, t3, t4, t5], 
        reviews=reviews, 
        scores=scores,
        rel_words=(",".join(words_to_highlight)),
        num_pages=(math.ceil(num_reviews / REVIEWS_PER_PAGE)),
        cur_page=page_number
    )


@ app.route("/reviews")
def reviews_search():
    text = request.args.get("query")
    countries = request.args.get("countries")
    tags = request.args.get("tags")
    return sql_search(text, countries, tags)

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
