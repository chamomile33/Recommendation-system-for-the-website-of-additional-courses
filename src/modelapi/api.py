import logging
import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import modelapi.logger
from modelapi.config import config
from modelapi.processor import Processor

app = FastAPI()
HOME_PAGE = 'https://www.hse.ru/edu/dpo/'


def courseid_to_link(courseid):
    if courseid is None:
        return None
    return HOME_PAGE + str(courseid)


@app.get("/recomendation/")
def get_rec(userid: str, fields_num: int = 10, adv_perc: float = 0, bottom_perc: float = 0, as_link: bool = False):
    if app.processor.check_userid(userid):
        return JSONResponse(content="User not found", status_code=404)
    
    recommendation = app.processor.recommend(userid, fields_num, adv_perc, bottom_perc)
    if as_link:
        recommendation = [courseid_to_link(courseid) for courseid in recommendation]
    return JSONResponse(content=recommendation, status_code=200)


@app.get("/recently_viewed/")
def get_recently_viewed(userid: str, fields_num: int = 10, as_link: bool = False):
    if app.processor.check_userid(userid):
        return JSONResponse(content="User not found", status_code=404)
    
    viewed = app.processor._data_manager._client.query(
        f'''
            SELECT extract(current_location, '([0-9]+)')
            FROM raw_interactions
            WHERE user_id LIKE '%|{userid}|%|%'
            GROUP BY extract(current_location, '([0-9]+)')
            ORDER BY MAX(event_timestamp) DESC
            LIMIT {fields_num}
        '''
    )
    viewed = viewed.result_columns[0]
    if as_link:
        viewed = [courseid_to_link(courseid) for courseid in viewed]
    return JSONResponse(content=viewed, status_code=200)


@app.get("/viewed/")
def get_viewed(userid: str, fields_num: int = 0, as_link: bool = False):
    if fields_num != 0:
        return get_recently_viewed(userid, fields_num, as_link)
    
    if app.processor.check_userid(userid):
        return JSONResponse(content="User not found", status_code=404)
    
    interactions = app.processor._data_manager.get_interaction_csr_matrix()
    userid_in_csr = app.processor._data_manager._users_id_converter.get_forward(userid)

    import numpy as np

    user_interactions = np.where(interactions.getrow(userid_in_csr).getnnz(axis=0) > 0)[0]
    viewed = [app.processor._data_manager._courses_id_converter.get_backward(courseid_in_csr) for courseid_in_csr in user_interactions]
    if as_link:
        viewed = [courseid_to_link(courseid) for courseid in viewed]
    return JSONResponse(content=viewed, status_code=200)


@app.get("/recomendation_and_viewed/")
def get_rec_and_viewed(userid: str, rec_fields_num: int = 10, viewed_fields_count: int = 0,adv_perc: float = 0, bottom_perc: float = 0, as_link: bool = False):
    if app.processor.check_userid(userid):
        return JSONResponse(content="User not found", status_code=404)
    
    viewed_response = get_viewed(userid, viewed_fields_count, as_link)
    rec_response = get_rec(userid, rec_fields_num, adv_perc, bottom_perc, as_link)

    if viewed_response.status_code != 200:
        return viewed_response
    if rec_response.status_code != 200:
        return rec_response
    
    content = [
        json.loads(rec_response.body.decode("utf-8")),
        json.loads(viewed_response.body.decode("utf-8"))
    ]
    return JSONResponse(content=content, status_code=200)


@app.get("/user_course_coefficient/")
def read_item(userid: str, courseid: str):
    content = {"coefficient": app.processor._data_manager.get_user_course_coefficient(userid, courseid)}
    if content is None:
        return JSONResponse(content="User or course not found in system", status_code=404)
    return JSONResponse(content=content, status_code=200)


@app.get("/top_courses/")
def get_top(fields_num: int = 10, as_link: bool = False):
    courses = app.processor.get_top_courses(fields_num)
    if as_link:
        courses = [(courseid_to_link(courseid), n) for [courseid, n] in courses]
    return JSONResponse(content=courses, status_code=200)


@app.get("/bottom_courses/")
def get_bottom(fields_num: int = 10, as_link: bool = False):
    courses = app.processor.get_bottom_courses(fields_num)
    if as_link:
        courses = [(courseid_to_link(courseid), n) for [courseid, n] in courses]
    return JSONResponse(content=courses, status_code=200)


@app.get("/adv_courses/")
def get_adv(as_link: bool = False):
    courses = app.processor.get_adv_courses()
    if as_link:
        courses = [courseid_to_link(courseid) for courseid in courses]
    return JSONResponse(content=courses, status_code=200)


@app.get("/force_load_updates_and_full_retrain/")
async def force_retrain():
    app.scheduler.pause()
    temp, app.processor._last_full_fit_timestemp = app.processor._last_full_fit_timestemp, 0
    app.processor._full_incoming_updates = 1

    await app.processor.load_updates_and_retrain()
    if app.processor._last_full_fit_timestemp == 0:
        app.processor._last_full_fit_timestemp = temp
    app.scheduler.resume()

    return JSONResponse(content="Done", status_code=200)


@app.on_event("startup")
async def startup():
    app.logger = logging.getLogger('main')
    app.logger.info("Application initialization")

    app.processor = Processor()
    await app.processor.load_cosine_model()
    await app.processor.load_updates_and_retrain()

    app.scheduler = AsyncIOScheduler()
    app.scheduler.add_job(app.processor.load_updates_and_retrain, 'interval', seconds=int(config["jobs_threshold"]["load_updates"]))
    app.scheduler.start()

    app.logger.info("Application started")


@app.on_event("shutdown")
async def on_shutdown():
    app.scheduler.shutdown()