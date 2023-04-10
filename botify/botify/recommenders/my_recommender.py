from .random import Random
from .recommender import Recommender
import random
import numpy as np


W = 0.3
K = 100
T = 10


class MyRecomm(Recommender):
    """
    Recommend closest track for given user and his last track.
    1. Take context user embedding.
    2. Take context track embedding.
    3. Calculate mean embedding with weight W - mean_context_embed.
    4. Get 100 closest next tracks for context tracks (prev_track)
    6. Get 100 distances between mean_context_embed and closest next tracks and make probabilities from them
    7. Sample 1 track with such probabilities.
    """

    def __init__(self, app, tracks_redis, recom_for_tracks, catalog, track_context_embeds_redis, track_next_embeds_redis, user_context_embeds_redis, recom_for_users_redis):
        self.app = app
        self.recom_for_tracks = recom_for_tracks
        self.fallback = Random(tracks_redis)
        self.catalog = catalog
        self.track_context_embeds_redis = track_context_embeds_redis,
        self.track_next_embeds_redis = track_next_embeds_redis,
        self.user_context_embeds_redis = user_context_embeds_redis,
        self.recom_for_users_redis = recom_for_users_redis

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        #  достаем рекомендации для прослушанного трека
        my_nn_recomm = self.recom_for_tracks.get(prev_track)   # получаем предыдущий прослушанный трек из базы данных
        if my_nn_recomm is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        my_nn_recomm = self.catalog.from_bytes(my_nn_recomm)  # раскодируем
        closest_next_track_id_for_track = np.array(my_nn_recomm)

        #  получаем эмбеддинги ближайших треков к данному треку
        closest_next_track_embeds_for_track = []
        for i in closest_next_track_id_for_track:
            closest_next_track_embeds_for_track.append(self.catalog.from_bytes(self.track_next_embeds_redis[0].get(int(i))))
        closest_next_track_embeds_for_track = np.array(closest_next_track_embeds_for_track)

        #  получаем айди треков, ближайших к юзеру
        recomm_for_users = self.recom_for_users_redis.get(user)
        if recomm_for_users is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        recomm_for_users = self.catalog.from_bytes(recomm_for_users)
        closest_next_track_id_for_user = np.array(recomm_for_users)

        #  получаем эмбеддинги треков, ближайших к юзеру
        closest_next_track_embeds_for_user = []
        for i in closest_next_track_id_for_user:
            closest_next_track_embeds_for_user.append(
                self.catalog.from_bytes(self.track_next_embeds_redis[0].get(int(i))))
        closest_next_track_embeds_for_user = np.array(closest_next_track_embeds_for_user)

        # достаем эмбеддинг прослушанного трека
        context_track_row = self.track_context_embeds_redis[0].get(prev_track)
        if context_track_row is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        context_track_row = self.catalog.from_bytes(context_track_row)
        context_track_embed = np.array(context_track_row)

        #  достаем контекстный эмбеддинг пользователя
        context_user_row = self.user_context_embeds_redis[0].get(user)
        if context_user_row is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        context_user_row = self.catalog.from_bytes(context_user_row)
        context_user_embed = np.array(context_user_row)

        if prev_track_time < 0.5:
            mean_context_embed = context_user_embed
            next_tracks_emb = closest_next_track_embeds_for_user
            next_track_id = closest_next_track_id_for_user
        else:
            mean_context_embed = W * context_user_embed + (1 - W) * context_track_embed
            next_tracks_emb = closest_next_track_embeds_for_track
            next_track_id = closest_next_track_id_for_track

        distances = np.dot(next_tracks_emb, mean_context_embed)
        probabilities = np.exp(T * distances) / np.sum(np.exp(T * distances))
        sort_idxs = np.argsort(probabilities)
        idx_to_null = sort_idxs[:-K]

        for i in range(len(idx_to_null)):
            probabilities[idx_to_null[i]] = 0

        probabilities = probabilities / np.sum(probabilities)
        idx = np.random.choice(a=np.arange(len(probabilities)), p=probabilities)

        return int(next_track_id[idx])
