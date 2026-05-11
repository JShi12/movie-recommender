"""Shared feature-name contract used across pipeline stages."""

USER_ID_KEY = "user_id"
MOVIE_ID_KEY = "movie_id"
AGE_KEY = "age"
GENDER_KEY = "gender"
OCCUPATION_KEY = "occupation"
GENRES_KEY = "genres"
LABEL_KEY = "label"

USER_FEATURE_KEYS = [
    USER_ID_KEY,
    AGE_KEY,
    GENDER_KEY,
    OCCUPATION_KEY,
]

MOVIE_FEATURE_KEYS = [
    MOVIE_ID_KEY,
    GENRES_KEY,
]

