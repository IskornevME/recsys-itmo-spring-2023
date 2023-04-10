import random

from .random import Random
from .recommender import Recommender


class StickyArtist(Recommender):
    def __init__(self, tracks_redis, artists_redis, catalog):
        self.fallback = Random(tracks_redis)
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        track_data = self.tracks_redis.get(prev_track)  # берем инфу о треке из базы данных по айди
        if track_data is not None:
            track = self.catalog.from_bytes(track_data)  # десериализуем инфу
        else:
            raise ValueError(f"Track not found: {prev_track}")

        artist_data = self.artists_redis.get(track.artist)  # достаем инфу об исполнителе (по ключу исполнителя)
        if artist_data is not None:
            artist_tracks = self.catalog.from_bytes(artist_data)
        else:
            raise ValueError(f"Artist not found: {prev_track}")

        index = random.randint(0, len(artist_tracks) - 1)
        return artist_tracks[index]  # рекомендуем случайный трек исполнителя

