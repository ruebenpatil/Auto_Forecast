import numpy as np
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils.metrics import smape, evaluate_model

from src.utils.logger import setup_logger
import warnings

warnings.simplefilter("ignore")

logger = setup_logger("PROCESSING")

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()


def build_lstm_model(input_shape, n_units, dropout_rate=0.2, recurrent_dropout=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(n_units, activation="relu", return_sequences=True,
                   recurrent_dropout=recurrent_dropout, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(n_units // 2, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_lstm_model(X_train, X_test, y_train, y_test, n_trials=20):
    X_train = np.asarray(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.asarray(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    input_shape = (X_train.shape[1], 1)

    def objective(trial):
        n_units = trial.suggest_int("n_units", 32, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        recurrent_dropout = trial.suggest_float("recurrent_dropout", 0.0, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 30, 100)

        model = build_lstm_model(
            input_shape,
            n_units,
            dropout_rate=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            learning_rate=learning_rate
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        ]

        model.fit(X_train, y_train,
                  validation_split=0.2,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  verbose=0)

        preds = model.predict(X_test, verbose=0).flatten()
        return smape(y_test, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params

    # Train best model
    best_model = build_lstm_model(
        input_shape,
        n_units=best["n_units"],
        dropout_rate=best["dropout_rate"],
        recurrent_dropout=best["recurrent_dropout"],
        learning_rate=best["learning_rate"]
    )

    best_model.fit(X_train, y_train,
                   epochs=best["epochs"],
                   batch_size=best["batch_size"],
                   validation_split=0.2,
                   verbose=0,
                   callbacks=[
                       EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                       ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
                   ])

    # Baseline model (fixed config)
    baseline_model = build_lstm_model(input_shape, n_units=64)
    baseline_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    best_preds = best_model.predict(X_test, verbose=0).flatten()
    baseline_preds = baseline_model.predict(X_test, verbose=0).flatten()

    smape_best, mfb_best = evaluate_model(y_test, best_preds)
    smape_base, mfb_base = evaluate_model(y_test, baseline_preds)

    if smape_best < smape_base and abs(mfb_best) < abs(mfb_base):
        return best_preds
    else:
        return baseline_preds
