import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

try:
    from retrieval import config as project_config
except ImportError:
    try:
        import config as project_config
    except ImportError:
        project_config = None


def _config_value(name, default):
    return getattr(project_config, name, default) if project_config is not None else default


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

# Tunable defaults. The source of truth is retrieval/config.py; fallback values
# keep this TFX module loadable if a runner copies only trainer_module.py.
BATCH_SIZE = _config_value('BATCH_SIZE', 64)
EPOCHS = _config_value('EPOCHS', 10)
EARLY_STOPPING_PATIENCE = _config_value('EARLY_STOPPING_PATIENCE', 3)
LEARNING_RATE = _config_value('LEARNING_RATE', 0.001)
DROPOUT_RATE = _config_value('DROPOUT_RATE', 0.30)
L2_REGULARIZATION = _config_value('L2_REGULARIZATION', 1e-6)
USER_EMBEDDING_DIM = _config_value('USER_EMBEDDING_DIM', 32)
MOVIE_EMBEDDING_DIM = _config_value('MOVIE_EMBEDDING_DIM', 32)
AGE_EMBEDDING_DIM = _config_value('AGE_EMBEDDING_DIM', 8)
GENDER_EMBEDDING_DIM = _config_value('GENDER_EMBEDDING_DIM', 4)
OCCUPATION_EMBEDDING_DIM = _config_value('OCCUPATION_EMBEDDING_DIM', 12)
GENRE_EMBEDDING_DIM = _config_value('GENRE_EMBEDDING_DIM', 12)
FINAL_EMBEDDING_DIM = _config_value('FINAL_EMBEDDING_DIM', 32)

# Trainer custom_config keys.
BATCH_SIZE_KEY = 'batch_size'
EPOCHS_KEY = 'epochs'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience'
LEARNING_RATE_KEY = 'learning_rate'
DROPOUT_RATE_KEY = 'dropout_rate'
L2_REGULARIZATION_KEY = 'l2_regularization'
USER_EMBEDDING_DIM_KEY = 'user_embedding_dim'
MOVIE_EMBEDDING_DIM_KEY = 'movie_embedding_dim'
AGE_EMBEDDING_DIM_KEY = 'age_embedding_dim'
GENDER_EMBEDDING_DIM_KEY = 'gender_embedding_dim'
OCCUPATION_EMBEDDING_DIM_KEY = 'occupation_embedding_dim'
GENRE_EMBEDDING_DIM_KEY = 'genre_embedding_dim'
FINAL_EMBEDDING_DIM_KEY = 'final_embedding_dim'


def _as_int(value, default):
    if value is None:
        return default
    return int(value)


def _as_float(value, default):
    if value is None:
        return default
    return float(value)


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
        num_epochs=num_epochs, # None: repeat indefinitely
    )

    return dataset


def build_two_tower_model(
    user_vocab_size,
    movie_vocab_size,
    gender_vocab_size,
    occupation_vocab_size,
    genre_vocab_size,
    user_embedding_dim=USER_EMBEDDING_DIM,
    movie_embedding_dim=MOVIE_EMBEDDING_DIM,
    age_embedding_dim=AGE_EMBEDDING_DIM,
    gender_embedding_dim=GENDER_EMBEDDING_DIM,
    occupation_embedding_dim=OCCUPATION_EMBEDDING_DIM,
    genre_embedding_dim=GENRE_EMBEDDING_DIM,
    final_embedding_dim=FINAL_EMBEDDING_DIM,
    dropout_rate=DROPOUT_RATE,
    l2_regularization=L2_REGULARIZATION,
    learning_rate=LEARNING_RATE,
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
        output_dim=user_embedding_dim,
        name='user_embedding',
        embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization)
    )(user_id_input)
    user_embedding = layers.Flatten()(user_embedding)

    age_embedding = layers.Embedding(
        input_dim=AGE_BUCKETS + 1,
        output_dim=age_embedding_dim,
        name='age_embedding',
    )(age_input)
    age_embedding = layers.Flatten()(age_embedding)

    gender_embedding = layers.Embedding(
        input_dim=gender_vocab_size + 1,
        output_dim=gender_embedding_dim,
        name='gender_embedding',
    )(gender_input)
    gender_embedding = layers.Flatten()(gender_embedding)

    occupation_embedding = layers.Embedding(
        input_dim=occupation_vocab_size + 1,
        output_dim=occupation_embedding_dim,
        name='occupation_embedding',
    )(occupation_input)
    occupation_embedding = layers.Flatten()(occupation_embedding)

    user_features = layers.Concatenate()(
        [user_embedding, age_embedding, gender_embedding, occupation_embedding]
    )

    user_features = layers.Dropout(dropout_rate)(user_features)

    user_vector = layers.Dense(
        final_embedding_dim,
        activation='relu',
        name='user_tower',
    )(user_features)
    user_vector = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1),
        name='user_embedding_output',
    )(user_vector)

    # ===== MOVIE TOWER =====
    movie_embedding = layers.Embedding(
        input_dim=movie_vocab_size + 1,
        output_dim=movie_embedding_dim,
        name='movie_embedding',
        embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization)
    )(movie_id_input)
    movie_embedding = layers.Flatten()(movie_embedding)

    # Genres are already transformed into integer ids; normalize to ragged.
    genre_ragged = layers.Lambda(
        _to_genre_ragged,
        name='genre_to_ragged',
    )(genres_input)

    genre_embeddings = layers.Embedding(
        input_dim=genre_vocab_size + 1,
        output_dim=genre_embedding_dim,
        name='genre_embedding',
        embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization)
    )(genre_ragged)

    genre_features = layers.Lambda(
        lambda x: tf.math.divide_no_nan(
            tf.reduce_sum(x, axis=1),
            tf.cast(tf.expand_dims(x.row_lengths(), axis=1), x.dtype),
        ),
        name='genre_average_pooling',
    )(genre_embeddings)

    movie_features = layers.Concatenate()([movie_embedding, genre_features])
    movie_features = layers.Dropout(dropout_rate)(movie_features)

    movie_vector = layers.Dense(
        final_embedding_dim,
        activation='relu',
        name='movie_tower',
    )(movie_features)
    movie_vector = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1),
        name='movie_embedding_output',
    )(movie_vector)

    # ===== INTERACTION: DOT PRODUCT =====
    dot_product = layers.Dot(axes=1, name='dot_product')([user_vector, movie_vector])
    scaled_dot_product = layers.Lambda(
        lambda x: x * 5.0,
        name='scaled_dot_product',
    )(dot_product)
    output = layers.Activation('sigmoid', name='output')(scaled_dot_product)

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
    model.user_embedding_model = keras.Model(
        inputs={
            USER_ID_KEY: user_id_input,
            AGE_KEY: age_input,
            GENDER_KEY: gender_input,
            OCCUPATION_KEY: occupation_input,
        },
        outputs=user_vector,
        name='user_embedding_model',
    )
    model.movie_embedding_model = keras.Model(
        inputs={
            MOVIE_ID_KEY: movie_id_input,
            GENRES_KEY: genres_input,
        },
        outputs=movie_vector,
        name='movie_embedding_model',
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    return model


def run_fn(fn_args: FnArgs):
    """TFX Trainer run_fn."""

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    custom_config = getattr(fn_args, 'custom_config', None) or {}
    batch_size = _as_int(custom_config.get(BATCH_SIZE_KEY), BATCH_SIZE)
    epochs = _as_int(custom_config.get(EPOCHS_KEY), EPOCHS)
    early_stopping_patience = _as_int(
        custom_config.get(EARLY_STOPPING_PATIENCE_KEY),
        EARLY_STOPPING_PATIENCE,
    )
    learning_rate = _as_float(custom_config.get(LEARNING_RATE_KEY), LEARNING_RATE)
    dropout_rate = _as_float(custom_config.get(DROPOUT_RATE_KEY), DROPOUT_RATE)
    l2_regularization = _as_float(
        custom_config.get(L2_REGULARIZATION_KEY),
        L2_REGULARIZATION,
    )
    user_embedding_dim = _as_int(
        custom_config.get(USER_EMBEDDING_DIM_KEY),
        USER_EMBEDDING_DIM,
    )
    movie_embedding_dim = _as_int(
        custom_config.get(MOVIE_EMBEDDING_DIM_KEY),
        MOVIE_EMBEDDING_DIM,
    )
    age_embedding_dim = _as_int(custom_config.get(AGE_EMBEDDING_DIM_KEY), AGE_EMBEDDING_DIM)
    gender_embedding_dim = _as_int(
        custom_config.get(GENDER_EMBEDDING_DIM_KEY),
        GENDER_EMBEDDING_DIM,
    )
    occupation_embedding_dim = _as_int(
        custom_config.get(OCCUPATION_EMBEDDING_DIM_KEY),
        OCCUPATION_EMBEDDING_DIM,
    )
    genre_embedding_dim = _as_int(
        custom_config.get(GENRE_EMBEDDING_DIM_KEY),
        GENRE_EMBEDDING_DIM,
    )
    final_embedding_dim = _as_int(
        custom_config.get(FINAL_EMBEDDING_DIM_KEY),
        FINAL_EMBEDDING_DIM,
    )

    user_vocab_size = tf_transform_output.vocabulary_size_by_name(USER_ID_KEY)
    movie_vocab_size = tf_transform_output.vocabulary_size_by_name(MOVIE_ID_KEY)
    gender_vocab_size = tf_transform_output.vocabulary_size_by_name(GENDER_KEY)
    occupation_vocab_size = tf_transform_output.vocabulary_size_by_name(OCCUPATION_KEY)
    genre_vocab_size = tf_transform_output.vocabulary_size_by_name(GENRES_KEY)

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        batch_size=batch_size,
        num_epochs=None,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        batch_size=batch_size,
        num_epochs=None,
    )

    model = build_two_tower_model(
        user_vocab_size=user_vocab_size,
        movie_vocab_size=movie_vocab_size,
        gender_vocab_size=gender_vocab_size,
        occupation_vocab_size=occupation_vocab_size,
        genre_vocab_size=genre_vocab_size,
        user_embedding_dim=user_embedding_dim,
        movie_embedding_dim=movie_embedding_dim,
        age_embedding_dim=age_embedding_dim,
        gender_embedding_dim=gender_embedding_dim,
        occupation_embedding_dim=occupation_embedding_dim,
        genre_embedding_dim=genre_embedding_dim,
        final_embedding_dim=final_embedding_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
        learning_rate=learning_rate,
    )

    print('=' * 80)
    print('MODEL ARCHITECTURE')
    print('=' * 80)
    print('Genre pooling mode: average pooling')
    print(
        'Hyperparameters: '
        f'batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}, '
        f'dropout_rate={dropout_rate}, l2_regularization={l2_regularization}, '
        f'user_dim={user_embedding_dim}, movie_dim={movie_embedding_dim}, '
        f'genre_dim={genre_embedding_dim}, final_dim={final_embedding_dim}'
    )
    model.summary()

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=early_stopping_patience,
                restore_best_weights=True,
                mode='max',
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=1,
                mode='max',
                min_lr=1e-6,
            ),
            keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir),
        ],
    )

    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    transformed_feature_spec.pop(LABEL_KEY, None)
    user_feature_spec = {
        key: transformed_feature_spec[key]
        for key in [USER_ID_KEY, AGE_KEY, GENDER_KEY, OCCUPATION_KEY]
    }
    movie_feature_spec = {
        key: transformed_feature_spec[key]
        for key in [MOVIE_ID_KEY, GENRES_KEY]
    }

    def _parse_transformed_examples(serialized_examples, feature_spec, dense_keys):
        parsed = tf.io.parse_example(serialized_examples, feature_spec)

        # Model dense inputs are shape (batch, 1); parsed scalars are shape (batch,).
        for key in dense_keys:
            value = parsed[key]
            if not isinstance(value, tf.SparseTensor):
                parsed[key] = tf.expand_dims(tf.cast(value, tf.int64), axis=-1)
        return parsed

    # Explicit signatures accept serialized transformed tf.Example payloads.
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serving_default(serialized_examples):
        parsed = _parse_transformed_examples(
            serialized_examples,
            transformed_feature_spec,
            [USER_ID_KEY, MOVIE_ID_KEY, AGE_KEY, GENDER_KEY, OCCUPATION_KEY],
        )

        outputs = model(parsed, training=False)
        return {'outputs': outputs}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def user_embedding(serialized_examples):
        parsed = _parse_transformed_examples(
            serialized_examples,
            user_feature_spec,
            [USER_ID_KEY, AGE_KEY, GENDER_KEY, OCCUPATION_KEY],
        )
        user_inputs = {
            USER_ID_KEY: parsed[USER_ID_KEY],
            AGE_KEY: parsed[AGE_KEY],
            GENDER_KEY: parsed[GENDER_KEY],
            OCCUPATION_KEY: parsed[OCCUPATION_KEY],
        }
        outputs = model.user_embedding_model(user_inputs, training=False)
        return {'user_embedding': outputs}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def movie_embedding(serialized_examples):
        parsed = _parse_transformed_examples(
            serialized_examples,
            movie_feature_spec,
            [MOVIE_ID_KEY],
        )
        movie_inputs = {
            MOVIE_ID_KEY: parsed[MOVIE_ID_KEY],
            GENRES_KEY: parsed[GENRES_KEY],
        }
        outputs = model.movie_embedding_model(movie_inputs, training=False)
        return {'movie_embedding': outputs}

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={
            'serving_default': serving_default,
            'user_embedding': user_embedding,
            'movie_embedding': movie_embedding,
        },
    )
    print(f'Model saved to {fn_args.serving_model_dir}')
