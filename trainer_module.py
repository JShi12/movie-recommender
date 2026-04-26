import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# Feature names (must match transform module)
USER_ID_KEY = 'user_id'
MOVIE_ID_KEY = 'movie_id'
AGE_KEY = 'age'
GENDER_KEY = 'gender'
OCCUPATION_KEY = 'occupation'
GENRES_KEY = 'genres'
LABEL_KEY = 'label'

# Age buckets are fixed by transform config
AGE_BUCKETS = 6

# Embedding dimensions
USER_EMBEDDING_DIM = 32
MOVIE_EMBEDDING_DIM = 32
AGE_EMBEDDING_DIM = 12
GENDER_EMBEDDING_DIM = 6
OCCUPATION_EMBEDDING_DIM = 12
GENRE_EMBEDDING_DIM = 12
FINAL_EMBEDDING_DIM = 64


def _input_fn(file_pattern, tf_transform_output, batch_size=32, num_epochs=None):
    """Create input function for training/evaluation."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(
            filenames, compression_type='GZIP'
        ),
        label_key=LABEL_KEY,
        num_epochs=num_epochs,
    )

    return dataset


def build_two_tower_model(
    user_vocab_size,
    movie_vocab_size,
    gender_vocab_size,
    occupation_vocab_size,
    genre_vocab_size,
):
    """Build two-tower dot product model with embeddings."""

    def _to_genre_ragged(x):
        # Normalize sparse/dense/ragged genres to RaggedTensor for pooling.
        if isinstance(x, tf.SparseTensor):
            return tf.RaggedTensor.from_sparse(x)
        if isinstance(x, tf.RaggedTensor):
            return tf.cast(x, tf.int64)
        dense = tf.cast(x, tf.int64)
        sparse = tf.sparse.from_dense(dense)
        return tf.RaggedTensor.from_sparse(sparse)

    # ===== INPUT LAYERS =====
    user_id_input = layers.Input(shape=(1,), name=USER_ID_KEY, dtype=tf.int64)
    movie_id_input = layers.Input(shape=(1,), name=MOVIE_ID_KEY, dtype=tf.int64)
    age_input = layers.Input(shape=(1,), name=AGE_KEY, dtype=tf.int64)
    gender_input = layers.Input(shape=(1,), name=GENDER_KEY, dtype=tf.int64)
    occupation_input = layers.Input(shape=(1,), name=OCCUPATION_KEY, dtype=tf.int64)
    genres_input = layers.Input(shape=(None,), name=GENRES_KEY, dtype=tf.int64, sparse=True)

    # ===== USER TOWER =====
    user_embedding = layers.Embedding(
        input_dim=user_vocab_size + 1,
        output_dim=USER_EMBEDDING_DIM,
        name='user_embedding',
    )(user_id_input)
    user_embedding = layers.Flatten()(user_embedding)

    age_embedding = layers.Embedding(
        input_dim=AGE_BUCKETS + 1,
        output_dim=AGE_EMBEDDING_DIM,
        name='age_embedding',
    )(age_input)
    age_embedding = layers.Flatten()(age_embedding)

    gender_embedding = layers.Embedding(
        input_dim=gender_vocab_size + 1,
        output_dim=GENDER_EMBEDDING_DIM,
        name='gender_embedding',
    )(gender_input)
    gender_embedding = layers.Flatten()(gender_embedding)

    occupation_embedding = layers.Embedding(
        input_dim=occupation_vocab_size + 1,
        output_dim=OCCUPATION_EMBEDDING_DIM,
        name='occupation_embedding',
    )(occupation_input)
    occupation_embedding = layers.Flatten()(occupation_embedding)

    user_features = layers.Concatenate()(
        [user_embedding, age_embedding, gender_embedding, occupation_embedding]
    )

    user_vector = layers.Dense(
        FINAL_EMBEDDING_DIM,
        activation='relu',
        name='user_tower',
    )(user_features)
    user_vector = layers.LayerNormalization()(user_vector)

    # ===== MOVIE TOWER =====
    movie_embedding = layers.Embedding(
        input_dim=movie_vocab_size + 1,
        output_dim=MOVIE_EMBEDDING_DIM,
        name='movie_embedding',
    )(movie_id_input)
    movie_embedding = layers.Flatten()(movie_embedding)

    # Genres are already transformed into integer ids; normalize to ragged and mean-pool.
    genre_ragged = layers.Lambda(
        _to_genre_ragged,
        name='genre_to_ragged',
    )(genres_input)

    genre_embeddings = layers.Embedding(
        input_dim=genre_vocab_size + 1,
        output_dim=GENRE_EMBEDDING_DIM,
        name='genre_embedding',
    )(genre_ragged)

    genre_features = layers.Lambda(
        lambda x: tf.math.divide_no_nan(
            tf.reduce_sum(x, axis=1),
            tf.cast(tf.expand_dims(x.row_lengths(), axis=1), x.dtype),
        ),
        name='genre_features',
    )(genre_embeddings)

    movie_features = layers.Concatenate()([movie_embedding, genre_features])

    movie_vector = layers.Dense(
        FINAL_EMBEDDING_DIM,
        activation='relu',
        name='movie_tower',
    )(movie_features)
    movie_vector = layers.LayerNormalization()(movie_vector)

    # ===== INTERACTION: DOT PRODUCT =====
    dot_product = layers.Dot(axes=1, name='dot_product')([user_vector, movie_vector])
    output = layers.Activation('sigmoid', name='output')(dot_product)

    model = keras.Model(
        inputs={
            USER_ID_KEY: user_id_input,
            MOVIE_ID_KEY: movie_id_input,
            AGE_KEY: age_input,
            GENDER_KEY: gender_input,
            OCCUPATION_KEY: occupation_input,
            GENRES_KEY: genres_input,
        },
        outputs=output,
        name='two_tower_recommender',
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    return model


def run_fn(fn_args: FnArgs):
    """TFX Trainer run_fn."""

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    user_vocab_size = tf_transform_output.vocabulary_size_by_name(USER_ID_KEY)
    movie_vocab_size = tf_transform_output.vocabulary_size_by_name(MOVIE_ID_KEY)
    gender_vocab_size = tf_transform_output.vocabulary_size_by_name(GENDER_KEY)
    occupation_vocab_size = tf_transform_output.vocabulary_size_by_name(OCCUPATION_KEY)
    genre_vocab_size = tf_transform_output.vocabulary_size_by_name(GENRES_KEY)

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        batch_size=64,
        num_epochs=None,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        batch_size=64,
        num_epochs=1,
    )

    model = build_two_tower_model(
        user_vocab_size=user_vocab_size,
        movie_vocab_size=movie_vocab_size,
        gender_vocab_size=gender_vocab_size,
        occupation_vocab_size=occupation_vocab_size,
        genre_vocab_size=genre_vocab_size,
    )

    print('=' * 80)
    print('MODEL ARCHITECTURE')
    print('=' * 80)
    model.summary()

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=10,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                restore_best_weights=True,
                mode='max',
            ),
            keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir),
        ],
    )

    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    transformed_feature_spec.pop(LABEL_KEY, None)

    # Explicit signature for TFMA: accept serialized tf.Example and parse transformed features.
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serving_default(serialized_examples):
        parsed = tf.io.parse_example(serialized_examples, transformed_feature_spec)

        # Model dense inputs are shape (batch, 1); parsed scalars are shape (batch,).
        for key in [USER_ID_KEY, MOVIE_ID_KEY, AGE_KEY, GENDER_KEY, OCCUPATION_KEY]:
            value = parsed[key]
            if not isinstance(value, tf.SparseTensor):
                parsed[key] = tf.expand_dims(tf.cast(value, tf.int64), axis=-1)

        outputs = model(parsed, training=False)
        return {'outputs': outputs}

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={'serving_default': serving_default},
    )
    print(f'Model saved to {fn_args.serving_model_dir}')
