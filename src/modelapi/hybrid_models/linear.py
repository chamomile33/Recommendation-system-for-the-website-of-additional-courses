import operator
import os.path

from numpy import array as nparray
from json import load as jsonload
from implicit.als import AlternatingLeastSquares as ALS
from implicit.nearest_neighbours import BM25Recommender

class LinearHybrid:
    def __init__(self,model_collaborative,collaborative_params,model_content,content_params,a):
        self.a = a
        self.model_collaborative = model_collaborative
        self.model_content = model_content
        self.collaborative_params = collaborative_params
        self.content_params = content_params

    def recommend(
            self,userid
        , user_items
        , N = 10
        , filter_already_liked_items = True
        , filter_items = None
        , recalculate_user = False
        , items = None
    ):
        def count_score(a,collab,content):
            return collab*a + (1-a)*content
        
        def get_new_dict(a,collab_dict,content_dict):
            return {k: count_score(a,collab_dict.get(k, 0),content_dict.get(k, 0)) for k in set(collab_dict) | set(content_dict)}
        
        def get_sorted_dict(courses_dict,N):
            return dict(sorted(courses_dict.items(), key = operator.itemgetter(1),reverse=True)[:N])
        
        def get_keys_values(courses_dict):
            recs = [list(courses_dict[i].keys()) for i in range(len(courses_dict))]
            scores = [list(courses_dict[i].values()) for i in range(len(courses_dict))]
            return nparray(recs), nparray(scores)
        
        oversampled_num = min(5*N, user_items.shape[1])
        collab_recs,collab_scores = self.model_collaborative.recommend(userid, user_items, oversampled_num, filter_already_liked_items, filter_items, recalculate_user, items)
        content_recs,content_scores = self.model_content.recommend(userid, user_items, oversampled_num, filter_already_liked_items, filter_items, recalculate_user, items)
        courses_list_dicts_collab = [dict(zip(collab_recs[i],collab_scores[i])) for i in range(user_items.shape[0])]
        courses_list_dicts_content = [dict(zip(content_recs[i],content_scores[i])) for i in range(user_items.shape[0])]
        courses_list_dicts = [get_new_dict(self.a,courses_list_dicts_collab[i],courses_list_dicts_content[i]) for i in range(user_items.shape[0])]
        courses_list_dicts = [get_sorted_dict(courses_list_dicts[i],N) for i in range(user_items.shape[0])]
        return get_keys_values(courses_list_dicts)
    
    def fit(self,content_features, user_items, show_progress=True, callback=None):
        if(user_items != None):
            self.model_collaborative = self.model_collaborative.__class__(**self.collaborative_params)
            self.model_collaborative.fit(user_items,show_progress,callback)
        if(content_features != None):
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
        with open('src/modelapi/hybrid_models/linear_params.json') as json_file:
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
        
        return LinearHybrid(model_collaborative, collaborative_params, model_content, content_params, data["LinearHybrid"]["n"])
