import asyncio
import time
import logging
import scipy
import pickle
import os.path
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from modelapi.content_models.cosine import CosineContent
from modelapi.data_manager import DataManager
from modelapi.hybrid_models.linear import LinearHybrid
from modelapi.config import config

ModelType = LinearHybrid

class Processor:
    def __init__(self) -> None:
        self.__logger = logging.getLogger('processor')
        self._model_path = config['paths']['saved_models']
        self._content_model_path = config['paths']['saved_models'] + '/cosine.npz'
        self._processor_data_path = config['paths']['temp'] + '/processor_data.txt'
        self._interactions_path = config['paths']['temp'] + '/interactions.npz'

        self._last_full_fit_timestemp = 0
        self._last_part_fit_timestemp = 0
        self._part_incoming_updates = 0
        self._full_incoming_updates = 0

        self._data_manager = DataManager()

        self._model = ModelType.load(self._model_path)

        if os.path.exists(self._processor_data_path):
            with open(self._processor_data_path, 'rb') as f:
                data = pickle.load(f)
                self._last_full_fit_timestemp = data['last_full_fit_timestemp']
                self._last_part_fit_timestemp = data['last_part_fit_timestemp']
                self._part_incoming_updates = data['part_incoming_updates']
                self._full_incoming_updates = data['full_incoming_updates']

    async def load_updates_and_retrain(self):
        self.__logger.info("Start load and retrain")
        users_k = len(self._data_manager._changed_users)
        self._data_manager.load_updates()
        changed_users_k = len(self._data_manager._changed_users) - users_k
        self._full_incoming_updates += changed_users_k
        self._part_incoming_updates += changed_users_k

        if self._full_incoming_updates and time.time() > self._last_full_fit_timestemp + int(config["jobs_threshold"]["full_fit"]):
            self.__logger.info("Full retrain")

            await self.fit_in_another_process(is_full_fit=True)
            
            self._last_full_fit_timestemp = self._last_part_fit_timestemp = time.time()
            self._full_incoming_updates = 0

            self.__logger.info("Finished full retrain")
        elif self._part_incoming_updates and time.time() > self._last_part_fit_timestemp + int(config["jobs_threshold"]["part_fit"]):
            self.__logger.info("Part retrain")

            await self.fit_in_another_process(is_full_fit=False)

            self._last_part_fit_timestemp = time.time()
            self._part_incoming_updates = 0
        self.__logger.info("Finished load and retrain")

    def recommend(self, userid: int, N: int = 10, adv_perc = 0, bottom_perc = 0):
        interactions = self._data_manager.get_interaction_csr_matrix()
        userid_in_csr = self._data_manager._users_id_converter.get_forward(userid)
        userid_interactions = interactions.getrow(userid_in_csr)

        # TO DO FIX
        if userid_in_csr not in self._data_manager._changed_users:
            recommendation_in_csrids = self._model.recommend(userid=[userid_in_csr], user_items=userid_interactions, N=N, recalculate_user=False)[0][0]
        else:
            if userid_in_csr < interactions.shape[0]:
                recommendation_in_csrids = self._model.recommend(userid=[userid_in_csr], user_items=userid_interactions, N=N, recalculate_user=True)[0][0]
            else:
                recommendation_in_csrids = self._model.recommend(userid=[None], user_items=userid_interactions, N=N, recalculate_user=True)[0][0]

        courses_to_shuffle = set()
        if bottom_perc:
            bottom = self._data_manager._courses_top_list.most_common()[:-N-1:-1]
            similar = self._content_model.similar_items([recommendation_in_csrids[0]],int(N * bottom_perc),items=bottom)[0][0]
            courses_to_shuffle.update(set(similar))
        if adv_perc:
            adv = np.array(self._data_manager._adv_courses)
            similar = self._content_model.similar_items([recommendation_in_csrids[0]],int(N * adv_perc), items=adv)[0][0]
            courses_to_shuffle.update(set(similar))

        recommendation = [self._data_manager._courses_id_converter.get_backward(courseid) for courseid in recommendation_in_csrids]

        if len(courses_to_shuffle):
            courses_to_shuffle.difference_update(set(recommendation))
            courses_to_shuffle = list(courses_to_shuffle)
            random.shuffle(courses_to_shuffle)
            courses_to_shuffle = [self._data_manager._courses_id_converter.get_backward(courseid) for courseid in courses_to_shuffle]
            recommendation[-len(courses_to_shuffle):] = courses_to_shuffle
            
        return recommendation
    
    async def load_cosine_model(self):
        self._content_model = await CosineContent.load(self._data_manager, self._content_model_path)

    async def fit_in_another_process(self, is_full_fit = True):
        self.save_model()
        self.__logger.info("Models and interactions was saved")

        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, Processor.fit_from_file_to_file, is_full_fit, list(self._data_manager._changed_users), self._interactions_path, self._model_path, self._model_path)
            self.__logger.info("Finished fit from file to file")

        self._model = ModelType.load(self._model_path)
        self.__logger.info("Model was loaded")

    def save_model(self):
        self._model.save(self._model_path)
        with open(self._interactions_path, 'wb') as f:
            scipy.sparse.save_npz(f, self._data_manager.get_interaction_csr_matrix())

        with open(self._processor_data_path, 'wb') as f:
            pickle.dump({
                  "last_full_fit_timestemp": self._last_full_fit_timestemp 
                , "last_part_fit_timestemp": self._last_part_fit_timestemp
                , "part_incoming_updates": self._part_incoming_updates
                , "full_incoming_updates": self._full_incoming_updates
            }, f)

    def fit_from_file_to_file(is_full_fit, changed_users, interactions_path, model_path_from, model_path_to):
        with open(interactions_path, 'rb') as f:
            interactions = scipy.sparse.load_npz(f)
        model = ModelType.load(model_path_from)

        if is_full_fit:
            model.fit(interactions, interactions)
        else:
            model.partial_fit(changed_users, interactions[changed_users, :])

        model.save(model_path_to)

    def fit(self):
        interactions = self._data_manager.get_interaction_csr_matrix()
        self._model.fit(interactions, interactions)

    def partial_fit(self):
        interactions = self._data_manager.get_interaction_csr_matrix()
        self._model.partial_fit(self._changed_users, interactions[self._changed_users, :])

    def check_userid(self, userid: str):
        return self._data_manager._users_id_converter.get_forward(userid) is None
    
    def get_similar_items(self):
        pass
    
    def get_top_courses(self, N: int = 10):
        top = self._data_manager._courses_top_list.most_common(N)
        return [self._data_manager._courses_id_converter.get_backward(courseid) for [courseid, _] in top]
    
    def get_bottom_courses(self, N: int = 10):
        bottom = self._data_manager._courses_top_list.most_common()[:-N-1:-1]
        return [self._data_manager._courses_id_converter.get_backward(courseid) for [courseid, _] in bottom]
    
    def get_adv_courses(self):
        adv = self._data_manager._adv_courses
        return [self._data_manager._courses_id_converter.get_backward(courseid) for courseid in adv]