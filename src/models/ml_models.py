import optuna
from src.utils.metrics import smape

def run_ml_model(model, train, test, train_features, test_features, n_trials=10):
    model.fit(train_features, train)
    base_predictions = model.predict(test_features)
    base_smape = smape(test, base_predictions)

    def objective(trial):
        param_grid = {
            "RandomForestRegressor": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            },
            "XGBRegressor": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            },
            "LGBMRegressor": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            },
            "CatBoostRegressor": {
                "iterations": trial.suggest_int("iterations", 50, 300, step=50),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            },
            "SVR": {
                "C": trial.suggest_float("C", 0.1, 10),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "rbf", "poly"]
                ),
            },
            "KNeighborsRegressor": {
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
            },
        }

        model_class = model.__class__.__name__
        tuned_model = model.__class__(**param_grid.get(model_class, {}))
        tuned_model.fit(train_features, train)
        predictions = tuned_model.predict(test_features)
        return smape(test, predictions)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=120, show_progress_bar=True)

    best_params = study.best_params
    tuned_model = model.__class__(
        **{k: v for k, v in best_params.items() if k in model.get_params()}
    )
    tuned_model.fit(train_features, train)
    tuned_predictions = tuned_model.predict(test_features)

    return (
        tuned_predictions
        if smape(test, tuned_predictions) < base_smape
        else base_predictions
    )