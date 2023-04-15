import numpy as np

#this function comes from assignment a5
def rocchio(query, relevant, irrelevant, input_doc_matrix,
            movie_name_to_index, a=.3, b=.3, c=.8, clip=True):
    """Returns a vector representing the modified query vector. 

    Note: 
        If the `clip` parameter is set to True, the resulting vector should have 
        no negatve weights in it!

        Also, be sure to handle the cases where relevant and irrelevant are empty lists.

    Params: {query: String (the name of the movie being queried for),
             relevant: List (the names of relevant movies for query),
             irrelevant: List (the names of irrelevant movies for query),
             input_doc_matrix: Numpy Array,
             movie_name_to_index: Dict,
             a,b,c: floats (weighting of the original query, relevant queries,
                             and irrelevant queries, respectively),
             clip: Boolean (whether or not to clip all returned negative values to 0)}
    Returns: Numpy Array 
    """
    # get query vector corresponding to movie title
    q0 = input_doc_matrix[movie_name_to_index[query], :]
    # get average relevant document vector
    avg_rel = np.zeros(input_doc_matrix.shape[1])
    for n in relevant:
        avg_rel += input_doc_matrix[movie_name_to_index[n], :]
    if len(relevant) > 0:
        avg_rel /= len(relevant)
    # get average irrelevant document vector
    avg_irel = np.zeros(input_doc_matrix.shape[1])
    for n in irrelevant:
        avg_irel += input_doc_matrix[movie_name_to_index[n], :]
    if len(irrelevant) > 0:
        avg_irel /= len(irrelevant)
    q1 = (a * q0) + (b * avg_rel) - (c * avg_irel)
    q1[q1 < 0] = 0
    return q1
