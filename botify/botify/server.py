import json
import logging
import time
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.recommenders.toppop import TopPop
from botify.recommenders.indexed import Indexed
from botify.recommenders.contextual import Contextual
from botify.recommenders.my_recommender import MyRecomm
from botify.track import Catalog

import numpy as np

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

# TODO Seminar 6 step 3: Create redis DB with tracks with diverse recommendations
tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
recom_for_tracks_redis = Redis(app, config_prefix="REDIS_RECOM_FOR_TRACKS")
track_context_embeds_redis = Redis(app, config_prefix="REDIS_CONTEXT_TRACK_EMBEDS")  # контекстные эмбеддинги треков
track_next_embeds_redis = Redis(app, config_prefix="REDIS_NEXT_TRACK_EMBEDS")  # обычные эмбеддинги треков
user_context_embeds_redis = Redis(app, config_prefix="REDIS_CONTEXT_USER_EMBEDS")  # контекстные эмбеддинги пользователей
recom_for_users_redis = Redis(app, config_prefix="REDIS_RECOM_FOR_USERS")

data_logger = DataLogger(app)

# TODO Seminar 6 step 4: Upload tracks with diverse recommendations to redis DB
catalog = Catalog(app).load(
    "./data/tracks_with_recs.json", app.config["TOP_TRACKS_CATALOG"]  # для эксперимента c nn
)
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_recom_for_tracks(recom_for_tracks_redis.connection, app.config["RECOM_FOR_TRACKS_PATH"])
catalog.upload_cont_track_embeds(track_context_embeds_redis.connection, app.config["CONTEXT_TRACK_EMBEDS_PATH"])
catalog.upload_next_track_embeds(track_next_embeds_redis.connection, app.config["NEXT_TRACK_EMBEDS_PATH"])
catalog.upload_cont_user_embeds(user_context_embeds_redis.connection, app.config["CONTEXT_USER_EMBEDS_PATH"])
catalog.upload_recom_for_users(recom_for_users_redis.connection, app.config["RECOM_FOR_USERS_PATH"])

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
        }


class Track(Resource):
    def get(self, track: int):
        data = tracks_redis.connection.get(track)
        if data is not None:
            return asdict(catalog.from_bytes(data))
        else:
            abort(404, description="Track not found")


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()

        # TODO Seminar 6 step 6: Wire RECOMMENDERS A/B experiment
        treatment = Experiments.MY_MODEL.assign(user)
        if treatment == Treatment.T1:
            recommender = MyRecomm(
                app,
                tracks_redis.connection,
                recom_for_tracks_redis.connection,
                catalog,
                track_context_embeds_redis.connection,
                track_next_embeds_redis.connection,
                user_context_embeds_redis.connection,
                recom_for_users_redis.connection
            )
        else:
            recommender = Contextual(tracks_redis.connection, catalog)

        recommendation = recommender.recommend_next(user, args.track, args.time)

        data_logger.log(
            "next",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
                recommendation,
            ),
        )
        return {"user": user, "track": recommendation}


class LastTrack(Resource):
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()
        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
            ),
        )
        return {"user": user}


api.add_resource(Hello, "/")
api.add_resource(Track, "/track/<int:track>")
api.add_resource(NextTrack, "/next/<int:user>")
api.add_resource(LastTrack, "/last/<int:user>")


if __name__ == "__main__":
    http_server = WSGIServer(("", 5050), app)
    http_server.serve_forever()
