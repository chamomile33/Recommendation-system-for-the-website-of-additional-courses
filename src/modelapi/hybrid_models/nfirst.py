import os.path

from itertools import chain, zip_longest
from numpy import array as nparray
from json import load as jsonload
from implicit.als import AlternatingLeastSquares as ALS
from implicit.nearest_neighbours import BM25Recommender

class NfirstHybrid:
    def __init__(self, model_collaborative, collaborative_params, model_content, content_params, n):
        self.collaborative_params = collaborative_params
        self.model_collaborative = model_collaborative
        self.content_params = content_params
        self.model_content = model_content
        self.n = n

    def recommend(
          self
        , userid
        , user_items
        , N = 10
        , filter_already_liked_items = True
        , filter_items = None
        , recalculate_user = False
        , items = None
    ):

        def get_unique_recs(recs_main, recs_score, recs_2, number_of_recs):
            return [[item,recs_score[i]] for i,item in enumerate(recs_main) if item not in recs_2][:number_of_recs]
        
        def get_recs(recs_scores):
            return [item[0] for item in recs_scores]
        
        def get_scores(recs_scores):
            return [item[1] for item in recs_scores]

        def predicator(x):
            return x != None

        def merge_two_recs(collab, content):
            merged = list((chain.from_iterable(zip_longest(collab,content))))
            return [*filter(predicator, merged)]
        
        n_collab = max(int(self.n * N),1)

        collab = self.model_collaborative.recommend(
              userid
            , user_items
            , n_collab
            , filter_already_liked_items
            , filter_items
            , recalculate_user
            , items
        )
    
        content = self.model_content.recommend(
              userid
            , user_items
            , N
            , filter_already_liked_items
            , filter_items
            , recalculate_user
            , items
        )
        
        content = [get_unique_recs(content[0][i],content[1][i],collab[0][i],N-n_collab) for i in range(len(content[0]))]
        content_recs =  [get_recs(content[i]) for i in range(len(content))]
        content_scores =  [get_scores(content[i]) for i in range(len(content))]
        result_items = [merge_two_recs(collab[0][i], content_recs[i]) for i in range(len(content_recs))]
        
        result_scores = [merge_two_recs(collab[1][i], content_scores[i]) for i in range(len(content_scores))]
        return nparray(result_items), nparray(result_scores)
    
    def fit(self, content_features, user_items, show_progress=True, callback=None):
        if(user_items is not None):
            self.model_collaborative = self.model_collaborative.__class__(**self.collaborative_params)
            self.model_collaborative.fit(user_items,show_progress,callback)
        if(content_features is not None):
            self.model_content = self.model_content.__class__(**self.content_params)
            self.model_content.fit(content_features,show_progress,callback)

    def partial_fit(self, users, user_items):
        self.model_collaborative.partial_fit_users(users,user_items)

    def save(self, path):
        with open(path + "model_content.npz", 'wb') as f:
            self.model_content.save(f)
        with open(path + "model_collaborative.npz", 'wb') as f:
            self.model_collaborative.save(f)

    def load(path = None):
            with open('src/modelapi/hybrid_models/nfirst_params.json') as json_file:
                data = jsonload(json_file)

            collaborative_params = data["ALS"]
            model_collaborative = ALS(**collaborative_params)
            content_params = data["BM25Recommender"]
            model_content = BM25Recommender(**content_params)

            if path is not None:
                if os.path.exists(path + "model_content.npz"):
                    with open(path + "model_content.npz", 'rb') as f:
                        model_content = model_content.load(f)
                if os.path.exists(path + "model_collaborative.npz"):
                    with open(path + "model_collaborative.npz", 'rb') as f:
                        model_collaborative = model_collaborative.load(f)
            
            return NfirstHybrid(model_collaborative, collaborative_params, model_content, content_params, data["NfirstHybrid"]["n"])