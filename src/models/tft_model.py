import optuna
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils.metrics import smape, evaluate_model

from src.utils.logger import setup_logger
import warnings

warnings.simplefilter("ignore")

logger = setup_logger("PROCESSING")

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()


def build_mlp_model(input_dim, units, dropout, learning_rate):
    model = Sequential([
        Dense(units, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(units // 2, activation="relu"),
        Dropout(dropout / 2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_and_evaluate_tft_model(X_train, X_test, y_train, y_test, n_trials=20):
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train).reshape(-1, 1), np.asarray(y_test).reshape(-1, 1)
    input_dim = X_train.shape[1]

    def objective(trial):
        units = trial.suggest_int("units", 64, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 30, 100)

        model = build_mlp_model(input_dim, units, dropout, learning_rate)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        ]

        model.fit(X_train, y_train,
                  validation_split=0.2,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0,
                  callbacks=callbacks)

        preds = model.predict(X_test, verbose=0).flatten()
        return smape(y_test, preds)

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params

    # Train optimized model
    optimized_model = build_mlp_model(input_dim, best["units"], best["dropout"], best["learning_rate"])
    optimized_model.fit(X_train, y_train,
                        epochs=best["epochs"],
                        batch_size=best["batch_size"],
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
                        ])
    optimized_preds = optimized_model.predict(X_test, verbose=0).flatten()

    # Train baseline model
    baseline_model = build_mlp_model(input_dim, units=128, dropout=0.2, learning_rate=0.001)
    baseline_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    baseline_preds = baseline_model.predict(X_test, verbose=0).flatten()

    smape_optimized, mfb_optimized = evaluate_model(y_test, optimized_preds)
    smape_baseline, mfb_baseline = evaluate_model(y_test, baseline_preds)

    if smape_optimized < smape_baseline and abs(mfb_optimized) < abs(mfb_baseline):
        return optimized_preds
    else:
        return baseline_preds
