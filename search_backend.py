# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ I M P O R T S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import math
import sys
import re
# from inverted_index_gcp import MultiFileReader, InvertedIndex # change this line
import json
import string
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby, chain
import pandas as pd
import os
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from google.cloud import storage

nltk.download('stopwords')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ T A B L E    O F    C O N T E N T S  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~   1. Globals and Constants                                                                     ~~~~
#~~~~   2. Create Inverted Index                                                                     ~~~~
#~~~~       2.1 GCP                                                                                  ~~~~
#~~~~         2.1.1 inverted_body_with_stemming                                                      ~~~~
#~~~~         2.1.2 inverted_title_with_stemming                                                     ~~~~
#~~~~        2.1.3 DL                                                                                ~~~~
#~~~~         2.1.4 NF                                                                               ~~~~
#~~~~   3. Helper Functions                                                                          ~~~~
#~~~~       3.1 tokenization Query                                                                   ~~~~
#~~~~       3.2 return Top N doc With Titles                                                         ~~~~
#~~~~   4. Ranking Methods                                                                           ~~~~
#~~~~       4.1 Boolean                                                                              ~~~~
#~~~~         4.1.1 booleanRanking                                                                   ~~~~
#~~~~       4.2 BM25                                                                                 ~~~~
#~~~~         4.2.1 calc_idf                                                                         ~~~~
#~~~~         4.2.2 BM25_and_binary_search                                                           ~~~~                                                                ~~~~
#~~~~       4.3 Cosine Similarity                                                                    ~~~~
#~~~~         4.3.1 search_body_implement                                                            ~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ G L O B A L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create stopwords set
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

########################################################################################################################
##################################### C R E A T E    I N V E R T E D    I N D E X ######################################
########################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ G C P ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.environ["GCLOUD_PROJECT"] = "ir-ass3-84763"
bucket_name = 'project_ir_os'  # project_ir_os
client = storage.Client()
bucket = client.bucket(bucket_name)

# body with stemming index from bucket
index_src = "index_body_with_stemming.pkl"
blob_index = bucket.blob(f"postings_gcp_body_with_stemming/{index_src}")
pickel_in = blob_index.download_as_string()
inverted_body_with_stemming = pickle.loads(pickel_in)

# title with stemming index from bucket
index_src = "index_title_with_stemming.pkl"
blob_index = bucket.blob(f"postings_gcp_title_with_stemming/{index_src}")
pickel_in = blob_index.download_as_string()
inverted_title_with_stemming = pickle.loads(pickel_in)

# This will be usefully for the calculation of AVGDL (utilized in BM25)
# DL from bucket
dl_src = "DL.pkl"
blob_DL = bucket.blob(f"DL/{dl_src}")
pickel_in = blob_DL.download_as_string()
DL = pickle.loads(pickel_in)

# This will be usefully for the calculation
# NF from bucket
nf_src = "nf.pkl"
blob_nf = bucket.blob(f"NF/{nf_src}")
pickel_in = blob_nf.download_as_string()
NF = pickle.loads(pickel_in)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H E L P E R     F U N C T I O N S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def tokenizationQuery(query, doStemming=False):
    """
    Tokenize the query and return a list of tokens
    """
    # create stopwords set
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens_without_stopwords = [i for i in tokens if not i in all_stopwords]  # query without stop words
    if doStemming:
        # create stopwords set without stemmer
        stemmer = PorterStemmer()
        for token in range(len(tokens_without_stopwords)):
            tokens_without_stopwords[token] = stemmer.stem(tokens_without_stopwords[token])

    return tokens_without_stopwords


def returnTopNdocWithTitles(inverted_title, searchDictVal, N=None):
    """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        inverted_title:
        searchDictVal: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default, N = None

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """
    res = []
    if N is None:
        temp_res = list(sorted(searchDictVal.items(), key=lambda x: x[1], reverse=True))  # sort by score
    else:
        temp_res = list(sorted(searchDictVal.items(), key=lambda x: x[1], reverse=True))[:N]  # sort by score (Cosine Similarity)

    for doc_id in temp_res:
        try:
            title = inverted_title.TitlesOfDocs[doc_id[0]]
            res.append((doc_id[0], title))
        except:
            pass
    return res


def normalizeQuery(word_counter):
    """
    Normalize the query vector, calculate (1/|q|)
    """
    sumNorm = sum(term ** 2 for term in word_counter.values())
    return (1 / math.sqrt(sumNorm))


########################################################################################################################
############################################ R A N K I N G   M E T H O D S #############################################
########################################################################################################################

"""
    1. Boolean (Search title and anchor)
    2. MB25 combine with binary (Search) 
    3. Cosine Similarity (Search body)
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~  B O O L E A N  ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def booleanRanking(inverted_index, query):
    query_tokens = np.unique(tokenizationQuery(query, doStemming=False))
    ID_SCORE = {}

    try:
        for term in query_tokens:
            if term in inverted_index.df:
                pls = inverted_index.read_posting_list(term, bucket_name)
                for tup_id_tf in pls:
                    ID_SCORE[tup_id_tf[0]] = ID_SCORE.get(tup_id_tf[0], 0) + 1
    except:
        pass

    return ID_SCORE


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~  B M 2 5  ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_idf(list_of_tokens, index):
    """
    This function calculate the idf values according to the BM25 idf formula for each term in the query.

    Parameters:
    -----------
    query: list of token representing the query. For example: ['look', 'blue', 'sky']

    Returns:
    -----------
    idf: dictionary of idf scores. As follows:
                                                key: term
                                                value: bm25 idf score
    """
    idf = {}
    N = len(DL) # number of documents in the corpus
    for term in list_of_tokens:
        if term in index.df.keys():
            n_ti = index.df[term]
            idf[term] = math.log(1 + (N - n_ti + 0.5) / (n_ti + 0.5))
        else:
            pass
    return idf

def BM25_and_binary_search(inverted_body, inverted_title, query, k1=1.5, b=0.75, N=None):
    TITLE_WEIGHT = 0.8
    BODY_WEIGHT = 0.2
    score = {}
    tokens_without_stopwords = tokenizationQuery(query, doStemming=False)
    N = len(DL)
    AVGDL = sum(DL.values()) / N
    idf = calc_idf(tokens_without_stopwords, inverted_body)
    for token in tokens_without_stopwords:
        try:
            pls = dict(inverted_body.read_posting_list(token, bucket_name))[:3000]
        except:
            continue
        for doc_id, df in pls.items():
            if doc_id not in score.keys():
                score[doc_id] = 0
            tf = pls.get(doc_id, 0)

            numerator = idf[token] * tf * (k1 + 1)
            denominator = (tf + k1 * (1 - b + b * DL.get(doc_id) / AVGDL))
            try:
                score[doc_id] += numerator / denominator  # sum up the score for each term in the query
            except ZeroDivisionError:
                pass

    bodyVal = list(sorted(score.items(), key=lambda x: x[1], reverse=True))


    TITLE_SCORE = booleanRanking(inverted_title, query)
    titleVal = list(sorted(TITLE_SCORE.items(), key=lambda x: x[1], reverse=True))

    merge_res = {}
    # merge results of body and title:
    for i in range(100):
        if i < len(bodyVal):
            docBody, scoreBody = bodyVal[i]
            scoreBody = scoreBody / bodyVal[0][1]
            if merge_res.get(docBody, 0) == 0:
                merge_res[docBody] = scoreBody
            else:
                merge_res[docBody] = merge_res.get(docBody) * TITLE_WEIGHT + BODY_WEIGHT * scoreBody

    for i in range(100):
        if i < len(titleVal):
            docTitle, scoreTitle = titleVal[i]
            scoreTitle = scoreTitle / len(tokens_without_stopwords)
            if merge_res.get(docTitle, 0) == 0:
                merge_res[docTitle] = scoreTitle
            else:
                merge_res[docTitle] = merge_res.get(docTitle) * BODY_WEIGHT + TITLE_WEIGHT * scoreTitle

    return returnTopNdocWithTitles(inverted_title=inverted_title, searchDictVal=merge_res, N=100)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ C O S I N E    S I M I L A R I T Y  ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def search_body_implement(inverted_body, query):
    """
    Returns up to a 100 search results for the query using TFIDF AND COSINE
    SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
    staff-provided tokenizer from Assignment 3 (GCP part) to do the
    tokenization and remove stopwords.
    Args:
        inverted_body:
        query: string of query words separated by spaces.
    Returns:
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """

    query_tokens = tokenizationQuery(query, doStemming=False)
    word_counter = Counter(query_tokens)

    docRank = {}  # dictionary of (wiki_id, cosine similarity)
    NUMBER_OF_DOCS = inverted_body.numberOfDocs  # number of docs in the corpus is 6348910
    cosineSimilarity = {}

    try:
        for term in query_tokens:
            pls = inverted_body.read_posting_list(term, bucket_name)
            idf = math.log(NUMBER_OF_DOCS / inverted_body.df[term], 10)  # calculate idf

            # calculate tf-idf
            for tup_id_tf in pls:
                tf = tup_id_tf[1]  # TODO: how to normalize???? ask Nir for doc or feq word per query
                docRank[tup_id_tf[0]] = docRank.get(tup_id_tf[0],
                                                    0) + tf * idf  # calculate cosine similarity {id: tf-idf}

        normalize_query = normalizeQuery(word_counter)  # calculate (1/|q|)

        for doc, cosSin_score_tf in docRank.items():
            nfi = inverted_body.nf.get(doc, 0)
            cosineSimilarity[doc] = cosSin_score_tf * normalize_query * nfi
    except:
        pass
    return returnTopNdocWithTitles(inverted_title=inverted_body, searchDictVal=cosineSimilarity, N=100)

