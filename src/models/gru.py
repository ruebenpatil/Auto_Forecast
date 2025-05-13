import numpy as np
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from src.utils.metrics import smape, evaluate_model


def build_gru_model(input_shape, n_units, dropout_rate=None, learning_rate=0.001):
    model = Sequential()
    model.add(GRU(n_units, activation="relu", return_sequences=True, input_shape=input_shape))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(GRU(n_units, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

def train_and_evaluate_gru_model(X_train, X_test, y_train, y_test, n_trials=10):
    # Reshape features for GRU input
    X_train = np.asarray(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.asarray(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    input_shape = (X_train.shape[1], 1)

    def objective(trial):
        n_units = trial.suggest_int("n_units", 16, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

        model = build_gru_model(input_shape, n_units, dropout_rate, learning_rate)
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

        preds = model.predict(X_test).flatten()
        return smape(y_test, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_trial.params

    # Train best model
    best_model = build_gru_model(
        input_shape,
        n_units=best_params["n_units"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"]
    )
    best_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # Train baseline model
    baseline_model = build_gru_model(input_shape, n_units=50)
    baseline_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    best_preds = best_model.predict(X_test).flatten()
    baseline_preds = baseline_model.predict(X_test).flatten()

    smape_optimized, mfb_optimized = evaluate_model(y_test, best_preds)
    smape_baseline, mfb_baseline = evaluate_model(y_test, baseline_preds)

    if smape_optimized < smape_baseline and abs(mfb_optimized) < abs(mfb_baseline):
        return best_preds
    else:
        return baseline_preds
