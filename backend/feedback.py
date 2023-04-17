import numpy as np
from collections import defaultdict

# the relevant dict which maps queries to their relevant documents
relevant_set = defaultdict(set)
# the irrelevant dict which maps queries to their irrelevant documents
irrelevant_set = defaultdict(set)

# this function comes from assignment a5


def rocchio(query: np.ndarray, input_doc_matrix: np.ndarray,
            hotel_name_to_index: dict, a: float = .3, b: float = .3, c: float = .8, clip: bool = True) -> np.ndarray:
    """Returns a vector representing the modified query vector. 

    Note: 
        If the `clip` parameter is set to True, the resulting vector should have 
        no negatve weights in it!

        Also, be sure to handle the cases where relevant and irrelevant are empty lists.

    Params: {query: String (the name of the movie being queried for),
             input_doc_matrix: Numpy Array,
             movie_name_to_index: Dict,
             a,b,c: floats (weighting of the original query, relevant queries,
                             and irrelevant queries, respectively),
             clip: Boolean (whether or not to clip all returned negative values to 0)}
    Returns: Numpy Array 
    """
    relevant = relevant_set[query_to_key(query)]
    irrelevant = irrelevant_set[query_to_key(query)]
    # get query vector corresponding to movie title
    q0 = input_doc_matrix[hotel_name_to_index[query], :]
    # get average relevant document vector
    avg_rel = np.zeros(input_doc_matrix.shape[1])
    for n in relevant:
        avg_rel += input_doc_matrix[hotel_name_to_index[n], :]
    if len(relevant) > 0:
        avg_rel /= len(relevant)
    # get average irrelevant document vector
    avg_irel = np.zeros(input_doc_matrix.shape[1])
    for n in irrelevant:
        avg_irel += input_doc_matrix[hotel_name_to_index[n], :]
    if len(irrelevant) > 0:
        avg_irel /= len(irrelevant)
    q1 = (a * q0) + (b * avg_rel) - (c * avg_irel)
    q1[q1 < 0] = 0
    return q1


'''{query_to_key query} takes a query stores as a numpy array and converts it
to a tuple so it can be used as a key in a python dict. This conversion will 
fit the following form: [0,1,0,2] -> (0,1,0,2)
'''


def query_to_key(query: np.ndarray) -> tuple:
    return tuple(query)


'''
{mark_relevant query hotel} marks a hotel as relevant to a query
by updating the relevance dict.
'''


def mark_relevant(query: np.ndarray, hotel) -> None:
    key = query_to_key(query)
    relevant_set[key].add(hotel)


'''
{mark_not_relevant query hotel} marks a hotel as irrelevant to a query
by updating the irrelevance dict.
'''


def mark_not_relevant(query: np.ndarray, hotel) -> None:
    key = query_to_key(query)
    irrelevant_set[key].add(hotel)
