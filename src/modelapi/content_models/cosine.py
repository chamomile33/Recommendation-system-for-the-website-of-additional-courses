import asyncio
import scipy
import os.path
import logging
import numpy as np
import scipy.sparse as sp

from concurrent.futures import ProcessPoolExecutor
from implicit.nearest_neighbours import CosineRecommender

class CosineContent:
    __logger = logging.getLogger("CosineModel")
        
    def load_courses_features(data_manager):
        CosineContent.__logger.info('Start setup courses features')

        with open("datasets/courses_features.csv", "rb") as file:
            courses_ids_in_csr_ids_order = np.loadtxt(file, usecols=1, dtype=np.str_, delimiter="^", skiprows=1)
            file.seek(0)
            courses_features_with_embeddings = np.loadtxt(file, delimiter="^", skiprows=1)  
        np.vectorize(data_manager._courses_id_converter.add)((courses_ids_in_csr_ids_order).astype(np.int32))
        courses_features_with_embeddings = np.delete(courses_features_with_embeddings, [0, 1], 1)
        courses_features_with_embeddings = sp.csr_matrix(courses_features_with_embeddings)

        CosineContent.__logger.info('Finished setup courses features')
        return courses_features_with_embeddings

    def _fit_cos(path):
        with open("temp.npz", 'rb') as f:
            courses_features_with_embeddings = scipy.sparse.load_npz(f)

        item_item_model = CosineRecommender(K=131)
        item_item_model.fit(courses_features_with_embeddings,True)

        with open(path, 'wb') as f:
            item_item_model.save(f)

    async def load(data_manager, path):
        CosineContent.__logger.info("Start load CosineContent")
        if not os.path.exists(path):
            CosineContent.__logger.info("Start train CosineContent")

            with open("temp.npz", 'wb') as f:
                scipy.sparse.save_npz(f, CosineContent.load_courses_features(data_manager))
            
            with ProcessPoolExecutor() as executor:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, CosineContent._fit_cos, path)

            os.remove("temp.npz")

            CosineContent.__logger.info("Finished train CosineContent")

        with open(path, 'rb') as f:
            model = CosineRecommender(K=131).load(f)

        CosineContent.__logger.info("Finished load CosineContent")

        return model
