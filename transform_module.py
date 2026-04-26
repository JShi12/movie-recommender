
import tensorflow as tf
import tensorflow_transform as tft

# Feature names
USER_ID_KEY = 'user_id'
MOVIE_ID_KEY = 'movie_id'
AGE_KEY = 'age'
GENDER_KEY = 'gender'
OCCUPATION_KEY = 'occupation'
GENRES_KEY = 'genres'
LABEL_KEY = 'label'


def preprocessing_fn(inputs):
    """
    TFX Transform preprocessing function.
    
    Transforms raw features into model-ready features:
    - Creates vocabularies for categorical features
    - Bucketizes age
    - Processes genres
    """
    outputs = {}
    
    # User ID: Create vocabulary and map to integers
    outputs[USER_ID_KEY] = tft.compute_and_apply_vocabulary(
        inputs[USER_ID_KEY],
        vocab_filename=USER_ID_KEY,
        num_oov_buckets=1
    )
    
    # Movie ID: Create vocabulary and map to integers
    outputs[MOVIE_ID_KEY] = tft.compute_and_apply_vocabulary(
        inputs[MOVIE_ID_KEY],
        vocab_filename=MOVIE_ID_KEY,
        num_oov_buckets=1
    )
    
    # Age: Bucketize into 6 bins (quantile boundaries)
    outputs[AGE_KEY] = tft.bucketize(
        inputs[AGE_KEY],
        num_buckets=6,
        epsilon=0.01
    )
    # # Age: fixed buckets
    # tft.apply_buckets(
    # inputs[AGE_KEY],
    # bucket_boundaries=[18, 25, 35, 45, 55]
    # )
    
    # Gender: Create vocabulary (M/F)
    outputs[GENDER_KEY] = tft.compute_and_apply_vocabulary(
        inputs[GENDER_KEY],
        vocab_filename=GENDER_KEY,
        num_oov_buckets=1
    )
    
    # Occupation: Create vocabulary (21 categories)
    outputs[OCCUPATION_KEY] = tft.compute_and_apply_vocabulary(
        inputs[OCCUPATION_KEY],
        vocab_filename=OCCUPATION_KEY,
        num_oov_buckets=1
    )
    
    # Genres: Split pipe-delimited strings and learn/apply vocabulary from data
    genre_values = tf.reshape(inputs[GENRES_KEY], [-1])
    genre_tokens = tf.strings.split(genre_values, sep='|').to_sparse()
    outputs[GENRES_KEY] = tft.compute_and_apply_vocabulary(
        genre_tokens,
        vocab_filename=GENRES_KEY,
        num_oov_buckets=1
    )
    
    # Label: Binary (0/1)
    outputs[LABEL_KEY] = inputs[LABEL_KEY]
    
    return outputs
