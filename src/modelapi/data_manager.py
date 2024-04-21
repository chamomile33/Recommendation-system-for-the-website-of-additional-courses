import time
import clickhouse_connect
import re
import numpy as np
import scipy.sparse as sp
import logging
import os.path
from json import load as jsonload
from collections import Counter

from modelapi.config import sensitive_config
from modelapi.helpers import IdConverter

HOME_PAGE = 'https://www.hse.ru/edu/dpo/'
browser_events = ['start_session', 'fingerprint', 'tracker_created', 'dom_content_loaded']

logger = logging.getLogger('data_manager')


class DataManager():
    def __init__(self) -> None:
        self._client = clickhouse_connect.get_client(
              host=sensitive_config['clickhouse']['host']
            , port=int(sensitive_config['clickhouse']['port'])
            , username=sensitive_config['clickhouse']['username']
            , password=sensitive_config['clickhouse']['password']
        )
        self._last_read_time = 0
        
        self._changed_users = set()
        self._user_courses_interaction_info = dict()
        self._interaction_csr_matrix = sp.csr_matrix((0, 0), dtype=np.float32)

        self._users_id_converter = IdConverter()
        self._courses_id_converter = IdConverter()

        self._courses_top_list = Counter()
        
        self._adv_courses = list()
        if os.path.exists("adv_courses.json"):
            with open("adv_courses.json") as json_file:
                adv = jsonload(json_file)
            self._adv_courses = [self._courses_id_converter.add(courseid) for courseid in adv]
        

    def load_updates(self):
        logger.info('Start loading updates')

        with self._get_raw_updates_stream() as raw_stream:
            for raw_data in raw_stream:
                logger.info('Trying to update the data with a block of size ' + str(len(raw_data)))

                transformed_data = self._tranform_raw_data(raw_data)
                self._recalculate_interactions(transformed_data)
                
                def add_pop(x):
                    self._courses_top_list[x] += 1
                
                np.vectorize(add_pop)(transformed_data[:, 1])

                logger.info('Finished updating data with the block')

        dok_matrix = sp.dok_matrix(
            (
                  self._users_id_converter.get_count()
                , self._courses_id_converter.get_count()
            )
            , dtype=np.float32
        )

        for (user_id, course_id), info in self._user_courses_interaction_info.items():
            dok_matrix[user_id, course_id] = info.coefficient

        self._interaction_csr_matrix = dok_matrix.tocsr()

        logger.info('Finished loading updates')
    
    def get_user_course_coefficient(self, userid, courseid):
        userid_in_csr = self._users_id_converter.get_forward(userid)
        courseid_in_csr = self._courses_id_converter.get_forward(courseid)
        if userid_in_csr is None or courseid_in_csr is None:
            return None
        return self._user_courses_interaction_info.get((userid_in_csr, courseid_in_csr), self._UserCourseInteractionInfo()).coefficient
    
    def get_interaction_csr_matrix(self):
        return self._interaction_csr_matrix

    class _UserCourseInteractionInfo():
        def __init__(self) -> None:
            self.events_num = 0
            self.coefficient = 0

            self.anchors = set()
            self.has_submit_form_event = False
        
    def _get_raw_updates_stream(self):
        current_time = int(time.time())

        logger.info('Sending a query to clickhouse')
        raw_stream = self._client.query_np_stream(
            f'''
                SELECT event_name, current_location, user_id
                FROM raw_interactions
                WHERE event_timestamp >= {self._last_read_time}
                AND event_timestamp < {current_time}
            '''
        )
        logger.info('Received a response from clickhouse')

        self._last_read_time = current_time
        return raw_stream
    
    def _clear_link(x):
        x = x.split('?')[0]
        x = x.split(':~:text')[0]
        x = re.sub('/#', '#', x)
        x = re.sub('[#/]$', '', x)
        x = re.sub('%23', '#', x)
        return x

    _clear_links_column = np.vectorize(_clear_link)

    def _tranform_raw_data(self, raw_data):
        raw_data[:, 1] = self._clear_links_column(raw_data[:, 1])

        def get_anchor(link):
            arr = link.split(sep='#')
            arr.append(None)
            return arr[1]

        def get_course_id_from_link(link: str):
            if link.find(HOME_PAGE) != 0:
                return None
            start = len(HOME_PAGE)
            end = start
            while end != len(link) and link[end].isnumeric():
                end += 1
            return link[start:end]

        raw_data[:, 2] = np.vectorize(lambda x: x.split('|')[1])(raw_data[:, 2])
        raw_data = raw_data[~np.any((raw_data == 'nan') | (raw_data == 'none') | (raw_data == None), axis=1)]

        raw_data = np.concatenate((raw_data, np.vectorize(get_anchor)(raw_data[:, 1]).reshape(-1, 1)), axis=1)
        raw_data[:, 1] = np.vectorize(get_course_id_from_link)(raw_data[:, 1])

        raw_data[:, 2] = np.vectorize(self._users_id_converter.add)(raw_data[:, 2]).astype(np.int32)
        raw_data[:, 1] = np.vectorize(self._courses_id_converter.add)(raw_data[:, 1]).astype(np.int16)

        return raw_data
    
    def _recalculate_interactions(self, transformed_data):
        for [event, courseid, userid, anchor] in transformed_data:
            key = (userid, courseid)
            info = self._user_courses_interaction_info.get(key, self._UserCourseInteractionInfo())
            
            info.events_num += 1 if event not in browser_events else 0

            if anchor is not None:
                info.anchors.add(anchor)

            if 'submit_form' == event or 'form_submit' == event:
                info.has_submit_form_event = True

            new_coefficient = self._calculate_coefficient(info)

            if info.coefficient != new_coefficient:
                info.coefficient = new_coefficient
                self._changed_users.add(userid)
                self._user_courses_interaction_info[key] = info

    def _calculate_coefficient(self, info):
        coefficient = 0.08 + int(info.has_submit_form_event)
        coefficient += min(0.2, np.log(info.events_num + 1) * 0.4 / np.log(100000))
        coefficient += min(0.4, np.log(len(info.anchors) + 1) / np.log(1000))
        return coefficient
