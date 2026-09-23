"""Baseline banali: previsioni senza nessun modello e nessun addestramento.

    python baseline.py
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from cesnet_tszoo.configs import TimeBasedConfig

# carica_config/apri_dataset arrivano da task.py: stessa identica risoluzione di data-root e
# stessa apertura del dataset del modello centralizzato, cosi' baseline e modello non possono
# mai finire a leggere dati diversi.
from task import apri_dataset, carica_config


def metriche(veri, previsti):
    """mse, rmse, r2, mae calcolate su tutte le finestre insieme.

    """
    veri, previsti = veri.ravel(), previsti.ravel()
    mse = ((previsti - veri) ** 2).mean()
    rmse = root_mean_squared_error(veri, previsti)
    r2 = r2_score(veri, previsti)
    mae = mean_absolute_error(veri, previsti)
    return mse, rmse, r2, mae


def carica_finestre_di_test():
    """Apre CESNET-TimeSeries24 e restituisce le finestre di test di TUTTE le istituzioni valide,
"""
    config = carica_config()
    dataset = apri_dataset(config)
    dataset.set_dataset_config_and_initialize(
        TimeBasedConfig(
            ts_ids=dataset.get_available_ts_indices()["id_institution"].tolist(),  # tutte, prima del filtro NaN
            # split cronologico 70/15/15: train piu' vecchio, poi validation, poi test piu' recente
            train_time_period=float(config["train-time-period"]),
            val_time_period=float(config["val-time-period"]),
            test_time_period=float(config["test-time-period"]),
            features_to_take=[str(config["target-feature"])],
            sliding_window_size=int(config["training-window-size"]),        # 168 ore di input
            sliding_window_prediction_size=int(config["prediction-window-size"]),  # 24 ore da predire
            sliding_window_step=int(config["prediction-window-size"]),
            random_state=int(config["random-state"]),
            transform_with="min_max_scaler",  # fittato solo sul periodo di train
            nan_threshold=float(config["nan-threshold"]),  # scarta le istituzioni con troppi NaN
            include_ts_id=False,
            include_time=False,
        ),
        display_config_details=None,
    )

    istituzioni = sorted(dataset.dataset_config.ts_ids.tolist())  # quelle sopravvissute al filtro NaN
    print(f"Istituzioni: {len(istituzioni)}")

    # Accumula le finestre di ogni istituzione nella stessa lista: niente di per-client qui,
    # il centralizzato (e questa baseline) valutano tutte le istituzioni insieme in un blocco solo.
    X, Y = [], []
    for istituzione in istituzioni:
        for finestra_x, finestra_y in dataset.get_test_dataloader(ts_id=istituzione):
            X.append(finestra_x)
            Y.append(finestra_y)

    X, Y = np.concatenate(X), np.concatenate(Y)  # (n_finestre_totale, 168 o 24, 1 feature)
    return X[:, :, 0], Y[:, :, 0]                # via l'ultimo asse: c'e' una sola feature (il target)


def main():
    X, Y = carica_finestre_di_test()
    orizzonte = Y.shape[1]  # 24 ore da predire

    # Ogni previsione copia valori gia' presenti in X, la finestra delle ore osservate:
    #   X[:, -1]           l'ultima ora osservata
    #   X[:, -orizzonte:]  le ultime `orizzonte` ore osservate
    #   X[:, :orizzonte]   le prime ore osservate, cioe' l'inizio della finestra

    finestra = X.shape[1]
    previsioni = {
        "ora precedente": np.repeat(X[:, -1:], orizzonte, axis=1),
        f"stessa ora, {orizzonte}h prima": X[:, -orizzonte:],
        f"stessa ora, {finestra}h prima (inizio finestra)": X[:, :orizzonte],
    }

    print(f"\nBaseline sul test set globale: {len(Y)} finestre, {orizzonte} ore predette\n")
    for nome, previsti in previsioni.items():
        mse, rmse, r2, mae = metriche(Y, previsti)
        print(f"  {nome:26s} mse={mse:.5f} rmse={rmse:.4f} r2={r2:.4f} mae={mae:.4f}")


if __name__ == "__main__":
    main()
