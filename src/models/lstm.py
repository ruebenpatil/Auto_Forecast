import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna

from src.utils.metrics import smape


def train_lstm_model(X_train, X_test, y_train, y_test, n_trials=10):
    X_train = np.asarray(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.asarray(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    def build_model(n_units, dropout_rate, learning_rate):
        model = Sequential([
            LSTM(n_units, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(dropout_rate),
            LSTM(n_units, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model
    
    def objective(trial):
        n_units = trial.suggest_int("n_units", 16, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)

        model = build_model(n_units, dropout_rate, learning_rate)
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=16, callbacks=[early_stopping], verbose=0)

        predictions = model.predict(X_test).flatten()
        return smape(y_test, predictions)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = n_trials)

    # Train best model
    best_params = study.best_trial.params
    final_model = build_model(best_params["n_units"], best_params["dropout_rate"], best_params["learning_rate"])
    final_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    final_predictions = final_model.predict(X_test, verbose=0).flatten()
    return final_predictions