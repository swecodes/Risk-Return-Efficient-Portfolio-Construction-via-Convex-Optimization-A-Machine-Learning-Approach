import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, Bidirectional,
    Input, Concatenate, Flatten, Softmax, Multiply,
    LayerNormalization, MultiHeadAttention, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train, X_val, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, model.predict(X_val), model.predict(X_test)


def train_xgboost(X_train, y_train, X_val, X_test,
                  n_estimators=300, learning_rate=0.05,
                  max_depth=5, subsample=0.8,
                  colsample_bytree=0.8, random_state=42):
    base_xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        objective='reg:squarederror'
    )
    model = MultiOutputRegressor(base_xgb)
    model.fit(X_train, y_train)
    return model, model.predict(X_val), model.predict(X_test)

def _early_stop(patience=15, min_delta=1e-5):
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True
    )


def _reduce_lr(patience=8, factor=0.5, min_lr=1e-6, min_delta=1e-5):
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience,
        min_delta=min_delta,
        min_lr=min_lr,
        verbose=1
    )
#lstm  
def build_lstm(window, n_features, n_outputs, units=64, dropout=0.3):
    model = Sequential([
        LSTM(units, input_shape=(window, n_features), return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(n_outputs)
    ], name="LSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


def train_lstm(X_tr, y_tr, X_vl, y_vl, X_ts,
               window, n_features, n_outputs,
               epochs=150, batch_size=32):
    model = build_lstm(window, n_features, n_outputs)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[_early_stop()],
        verbose=0
    )
    print(f"LSTM stopped at epoch {len(history.history['loss'])}")
    return model, history, model.predict(X_vl), model.predict(X_ts)

#bilstm
def build_bilstm(window, n_features, n_outputs, units=32, dropout=0.5):
    model = Sequential([
        Bidirectional(LSTM(
            units,
            input_shape=(window, n_features),
            kernel_regularizer=l2(1e-3),
            recurrent_regularizer=l2(1e-3)
        )),
        Dropout(dropout),
        Dense(n_outputs)
    ], name="BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='mse'
    )
    return model


def train_bilstm(X_tr, y_tr, X_vl, y_vl, X_ts,
                 window, n_features, n_outputs,
                 epochs=150, batch_size=32):
    model = build_bilstm(window, n_features, n_outputs)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[_early_stop()],
        verbose=0
    )
    print(f"BiLSTM stopped at epoch {len(history.history['loss'])}")
    return model, history, model.predict(X_vl), model.predict(X_ts)

#cnn-lstm with multi scale attention mechanism 
class AttentionReduce(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, use_bias=False)
        self.softmax     = Softmax(axis=1)
        self.multiply    = Multiply()

    def call(self, x):
        scores  = self.score_dense(x)
        weights = self.softmax(scores)
        context = self.multiply([x, weights])
        return tf.keras.ops.sum(context, axis=1)


def build_cnnlstm(window, n_features, n_outputs,
                  conv_filters=(64, 32), lstm_units=(128, 64),
                  kernel_sizes=(3, 5, 10)):
    inp = Input(shape=(window, n_features), name="input")

    branches = []
    for k in kernel_sizes:
        x = Conv1D(conv_filters[0], kernel_size=k, padding='causal',
                   activation='relu', name=f"conv1_k{k}")(inp)
        x = Conv1D(conv_filters[1], kernel_size=k, padding='causal',
                   activation='relu', name=f"conv2_k{k}")(x)
        branches.append(x)

    merged = Concatenate(axis=-1, name="merge")(branches)
    merged = Dropout(0.2, name="drop_conv")(merged)

    x = LSTM(lstm_units[0], return_sequences=True,
             kernel_regularizer=l2(1e-4),
             recurrent_regularizer=l2(1e-4), name="lstm1")(merged)
    x = Dropout(0.3, name="drop_lstm1")(x)
    x = LSTM(lstm_units[1], return_sequences=True,
             kernel_regularizer=l2(1e-4), name="lstm2")(x)
    x = Dropout(0.3, name="drop_lstm2")(x)

    ctx = AttentionReduce(name="attn")(x)
    x   = Dense(32, activation='relu', name="dense1")(ctx)
    out = Dense(n_outputs, name="output")(x)

    model = Model(inp, out, name="CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='mse'
    )
    return model


def train_cnnlstm(X_tr, y_tr, X_vl, y_vl, X_ts,
                  window, n_features, n_outputs,
                  epochs=300, batch_size=64):
    model = build_cnnlstm(window, n_features, n_outputs)
    callbacks = [
        _early_stop(patience=20),
        _reduce_lr(patience=8),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print(f"CNN-LSTM stopped at epoch {len(history.history['loss'])}")
    return model, history, model.predict(X_vl), model.predict(X_ts)

#sacre
def build_sacre(window, n_feat_total, n_stocks,
                conv_units=(32, 16), lstm_units=32,
                n_heads=2, key_dim=16):

    n_feat_per_stock = n_feat_total // n_stocks

    inp = Input(shape=(window, n_feat_total), name="joint_input")

    stock_representations = []
    for i in range(n_stocks):
        s = i * n_feat_per_stock
        e = (i + 1) * n_feat_per_stock

        stock_feat = Lambda(
            lambda x, s=s, e=e: x[:, :, s:e],
            name=f"slice_{i}"
        )(inp)

        x = Conv1D(conv_units[0], kernel_size=5, padding='causal',
                   activation='relu', name=f"cnn1_{i}")(stock_feat)
        x = Conv1D(conv_units[1], kernel_size=3, padding='causal',
                   activation='relu', name=f"cnn2_{i}")(x)
        x = LSTM(lstm_units, return_sequences=False,
                 kernel_regularizer=l2(1e-4), name=f"lstm_{i}")(x)
        x = Dropout(0.2, name=f"drop_{i}")(x)
        stock_representations.append(x)

    stacked = Lambda(
        lambda xs: tf.stack(xs, axis=1),
        name="stack_stocks"
    )(stock_representations)

    attn_out = MultiHeadAttention(
        num_heads=n_heads, key_dim=key_dim, dropout=0.1,
        name="cross_stock_attn"
    )(stacked, stacked)

    attn_out = LayerNormalization(name="attn_norm")(attn_out + stacked)

    x      = Dense(16, activation='relu', name="score_dense")(attn_out)
    scores = Dense(1, name="score_out")(x)
    scores = Flatten(name="scores_flat")(scores)

    return Model(inp, scores, name="SACRE")


def _listnet_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    p_true = tf.nn.softmax(y_true, axis=-1)
    p_pred = tf.nn.softmax(y_pred, axis=-1)
    return -tf.reduce_mean(
        tf.reduce_sum(p_true * tf.math.log(p_pred + 1e-10), axis=-1)
    )


def train_sacre(X_tr, y_tr, X_vl, y_vl, X_ts,
                window, n_feat_total, n_stocks,
                epochs=150, batch_size=32, lr=1e-3):
    model = build_sacre(window, n_feat_total, n_stocks)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=_listnet_loss
    )
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    print(f"SACRE stopped at epoch {len(history.history['loss'])}")
    return model, history, model.predict(X_vl), model.predict(X_ts)
