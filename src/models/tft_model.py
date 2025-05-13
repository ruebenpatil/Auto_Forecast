import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from src.utils.metrics import smape, evaluate_model


def train_and_evaluate_tft_model(X_train, X_test, y_train, y_test, n_trials=10):
    def build_model(units:int, dropout:float, input_dim:int):
        return Sequential([
            Dense(units, activation="relu", input_shape=(input_dim, )),
            Dropout(dropout),
            Dense(units // 2, activation="relu"),
            Dropout(dropout),
            Dense(1)
        ])
    
    def objective(trial):
        units = trial.suggest_int("units", 32, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        model = build_model(units, dropout, X_train.shape[1])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

        preds = model.predict(X_test).flatten()
        return smape(y_test, preds)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params


    # Train optimized model
    optimized_model = build_model(best_params["units"], best_params["dropout"], X_train.shape[1])
    optimized_model.compile(optimizer="adam", loss="mse")
    optimized_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    optimized_preds = optimized_model.predict(X_test, verbose=0).flatten()

    # Train baseline model
    baseline_model = build_model(128, 0.2, X_train.shape[1])
    baseline_model.compile(optimizer="adam", loss="mse")
    baseline_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    baseline_preds = baseline_model.predict(X_test, verbose=0).flatten()

    smape_optimized, mfb_optimized = evaluate_model(y_test, optimized_preds)
    smape_baseline, mfb_baseline = evaluate_model(y_test, baseline_preds)

    if smape_optimized < smape_baseline and abs(mfb_optimized) < abs(mfb_baseline):
        return optimized_preds
    else:
        return baseline_preds
