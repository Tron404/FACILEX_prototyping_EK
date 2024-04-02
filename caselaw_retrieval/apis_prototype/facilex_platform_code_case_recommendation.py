import json
import numpy as np

def cosine_similarity(query_embedding: np.ndarray, search_embedding: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between 2 arrays
    ---
    Parameters:
    * `query_embedding`: embedding of the user's query case
    * `search_embedding`: array of embeddings of the entire corpus (minus the query case)\n
    ---
    Return: a 1D array of similarity scores between the query and all other cases
    """

    # first normalize the arrays to unit length
    query_embedding /= np.linalg.norm(query_embedding)
    search_embedding = (search_embedding.T / np.linalg.norm(search_embedding, axis = 1))

    # then compute the normalized dot product
    return np.dot(query_embedding, search_embedding)[0]

def check_celex_status(query_celex: str|list|None, retrieved_case_celex: str|list|None) -> bool:
    """
    Check whether the CELEX IDs of 2 cases match
    ---
    Parameters:
    * `query_celex`: CELEX ID(s) of the user`s query case
    * `retrieved_case_celex`: CELEX ID(s) of a candidate similar case\n
    ---
    Return: boolean value based on whether `query_celex` is `None` or whether there are any shared CELEX IDs between the query and candidate case 
    """

    return (query_celex == None) or (len(set(query_celex) & set(retrieved_case_celex)) > 0)

def cosine_similarity_search(query_embedding: np.ndarray, search_space_embedding: np.ndarray, query_celex: str|list|None, search_space: list, top_k: int = 5) -> list:
    """
    Given the embedded representation of a query case and the embeddings of the cases through which to search,
    return the `top_k` most similar cases based on cosine similarity from most similar to least.
    ---
    Parameters:
    * `query_embedding`: embedding of the user's query case
    * `search_space_embedding`: array of embeddings of the entire corpus (minus the query case)
    * `query_celex`: CELEX ID(s) of the user's query case
    * `search_space`: list of all cases (minus the query) through which to search
    * `top_k`: number of similar cases to be retrieved\n
    ---
    Return: most similar cases (that are not the query case), sorted from most to least similar
    """

    cosine_scores: list = cosine_similarity(query_embedding, search_space_embedding)

    # include the similarity score inside each case json
    search_space: list = [{**data_entry, "similarity_score": score} for data_entry, score in zip(search_space, cosine_scores)]
    # sort the similarity scores from most to least similar and select the top 5 most similar cases
    best_matches: list = sorted(search_space, key = lambda x: x["similarity_score"], reverse = True)[:top_k]

    # only show the user cases which do have the same CELEX number; if there are no CELEX IDs in the query case, then simply return the most semantically similar cases
    best_matches = [retrieved_case for retrieved_case in best_matches if check_celex_status(query_celex, retrieved_case["euProvisions"])]
    
    return best_matches

def read_json_data(json_case_query_path: str, json_search_embedding_path: str, json_search_text_path: str) -> tuple[dict, list, list]:
    """
    Load in all of the json data from the given paths
    ---
    Input:
    * `json_case_query_path`: the json of the query case;
    * `json_search_embedding_path`: the json containing the entire search corpus of cases;
    * `json_search_text_path`: the json containing the embeddings of the search corpus.\n
    ---
    Return (as a tuple):
    * dictionary of the input query case from the user
    * list of all cases' embeddings
    * list of all cases without the query case
    """

    query_data: dict = json.load(open(json_case_query_path, "r"))
    search_embeddings_data: list = json.load(open(json_search_embedding_path, "r"))
    search_text_data: list = json.load(open(json_search_text_path, "r"))

    # remove query case from the search corpus
    search_text_data = [item for item in search_text_data if item["uniqueId"] != query_data["uniqueId"]]

    return query_data, search_embeddings_data, search_text_data

def get_embeddings_from_json(search_embeddings_data: list, query_data_uid: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get an np array of embeddings from the list of jsons
    ---
    Input:
    * `search_embeddings_data`: list of all cases' embeddings
    * `query_data_uid`: unique ID of the query case\n
    ---
    Return (as a tuple):
    * embedding of the query case
    * np array of all cases' embeddings, without the query case
    """
    query_embedding: np.ndarray = np.asarray([search_space_item["embedding"] for search_space_item in search_embeddings_data if search_space_item["uniqueId"] == query_data_uid])
    search_embedding: np.ndarray = np.asarray([np.asarray(search_space_item["embedding"]) for search_space_item in search_embeddings_data if search_space_item["uniqueId"] != query_data_uid])

    return query_embedding, search_embedding

if __name__ == "__main__":
    query_data, search_embeddings_data, search_text_data = read_json_data(json_case_query_path="input_query.json", json_search_embedding_path="corpus_embedded.json", json_search_text_path="corpus.json")

    query_embedding, search_embedding = get_embeddings_from_json(search_embeddings_data, query_data["uniqueId"])

    similar_cases: list = cosine_similarity_search(query_embedding, search_embedding, query_data["euProvisions"], search_text_data, top_k=5)

    json.dump(similar_cases, open("example_output.json", "w"), indent = 2)