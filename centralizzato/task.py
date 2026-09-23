""" CENTRALIZZATO su CESNET-TimeSeries24 

"""

import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import AgreggationType, SourceType

TARGET_FEATURE_INDEX = 0    # uso solo la feature target, quindi l'indice è 0, (modello univariato)

CARTELLA = Path(__file__).parent   # qui stanno config.toml e final_model.pt


def carica_config() -> dict:
    """Legge config.toml e restituisce la sezione [config].

    data-root viene risolto rispetto a CARTELLA (dove sta questo file), non alla directory da
    cui si lancia il comando: il progetto funziona identico su qualsiasi macchina e da qualsiasi
    cwd, senza percorsi assoluti. La cartella viene creata se non esiste, cosi' alla prima
    esecuzione su un clone pulito cesnet_tszoo ci scarica dentro il dataset.
    """
    with open(CARTELLA / "config.toml", "rb") as f:
        config = tomllib.load(f)["config"]
    data_root = (CARTELLA / config["data-root"]).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    config["data-root"] = str(data_root)
    return config


def get_device():
    """Device di calcolo: GPU se disponibile, altrimenti CPU"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMForecast(nn.Module):
    """LSTM per forecasting: prende in input una finestra di training e predice gli step della finestra di predizione"""

    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        # output_size = prediction_window_size
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,  # i dati in ingresso hanno forma (batch_size, seq_len, input_size)
            dropout=dropout,   # non applicabile con un solo layer (serve per ridurre overfitting, spegnendo dei nodi)
            bidirectional=True,  # Legge la finestra di input anche a ritroso (scelta del benchmark di riferimento).
        )

        # Bidirezionale: l'LSTM produce 2 * hidden_size feature per timestep
        self.fc = nn.Linear(hidden_size * 2, output_size)

    #prende un batch di finestre e produce le predizioni
    def forward(self, x):
        " x: (batch_size, seq_len, input_size) -> out: (batch_size, output_size)"
        # La LSTM legge la finestra due volte: una in avanti (ora 0 -> 167) e una all'indietro
        # (ora 167 -> 0). Vogliamo il riassunto di ENTRAMBE le letture a lettura completa, cioe'
        # DOPO che ciascuna ha visto tutte le 168 ore. h_n contiene proprio
        # questo: il risultato finale di ognuna delle due direzioni.
        out, (h_n, _) = self.lstm(x)
        return self.fc(torch.cat((h_n[-2], h_n[-1]), dim=1))


def build_model(config: dict):
    """Costruisce SOLO l'architettura, senza allenarla.
    Legge gli iperparametri dalla config e restituisce una LSTM nuova, con pesi casuali.
    Il seed viene fissato una volta sola in main.py, prima di chiamare questa funzione.
    """
    return LSTMForecast(
        input_size=1,  # modello univariato: una sola feature (target-feature)
        hidden_size=int(config["hidden-size"]),
        num_layers=int(config["num-layers"]),
        dropout=float(config["dropout"]),
        output_size=int(config["prediction-window-size"]),
    )


def apri_dataset(config: dict):
    """Apre il dataset CESNET-TimeSeries24.
    Non carica ancora i dati: serve poi
    """
    return CESNET_TimeSeries24.get_dataset(
        data_root=str(config["data-root"]),
        source_type=SourceType.INSTITUTIONS,
        aggregation=AgreggationType.AGG_1_HOUR,
        dataset_type="time_based",
        display_details=False,
    )


def concatena_finestre(loader):
    """Scorre un DataLoader di cesnet_tszoo (una finestra alla volta) e concatena
    tutte le finestre in due unici array: (n_finestre, window_size, 1)."""
    X_all, Y_all = [], []
    for X, Y in loader: #il loader mi da una finestra alla volta, le inglobo rispettivamente in un contenitore X e Y
        X_all.append(X)
        Y_all.append(Y)
    return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0) #concateno tutte le finestre in un unico array numpy (n_finestre, window_size, 1)


def dividi_in_batch(X, Y, batch_size):
    """Taglia X e Y in blocchi consecutivi da batch_size finestre (l'ultimo può essere più corto).
    Restituisce la lista di questi blocchi come coppie (X_batch, Y_batch)."""
    return [(X[i:i + batch_size], Y[i:i + batch_size]) for i in range(0, len(X), batch_size)]


def prepara_dataset_globale(config: dict):
    """Apre il dataset, lo configura, e restituisce TUTTE le finestre di TUTTE le istituzioni
    valide concatenate in un UNICO dataset per ciascuno split: e' il pool completo, mai diviso.

    Restituisce ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test)).
    """
    dataset = apri_dataset(config)
    # tutte le istituzioni del dataset, prima di ogni filtro: serve solo per contare quante
    # ne scarta il nan-threshold
    pool = dataset.get_available_ts_indices()["id_institution"].tolist()
    tsconfig = TimeBasedConfig(
        ts_ids=pool,    #lista di più isitituzioni
        train_time_period=float(config["train-time-period"]),
        val_time_period=float(config["val-time-period"]),
        test_time_period=float(config["test-time-period"]),
        features_to_take=[str(config["target-feature"])],
        sliding_window_size=int(config["training-window-size"]),
        sliding_window_prediction_size=int(config["prediction-window-size"]),
        sliding_window_step=int(config["prediction-window-size"]),
        random_state=int(config["random-state"]),
        transform_with="min_max_scaler",    # scaler viene fittato solo sul training set
        nan_threshold=float(config["nan-threshold"]),  # esclude istituzioni con troppi NaN (verificato indipendentemente su train/val/test)
       
        # Restiamo sul riempimento a 0 di default (default_values)
        # Entrambi hanno default True: senza, la libreria aggiunge id istituzione e tempo
        # come colonne extra e il modello univariato riceverebbe piu' di una feature.
        include_ts_id=False,
        include_time=False,
    )
    dataset.set_dataset_config_and_initialize(tsconfig, display_config_details=None)
    # Nessun campionamento casuale: il centralizzato lavora SEMPRE su tutte le istituzioni valide.
   
    pool_candidata = sorted(dataset.dataset_config.ts_ids.tolist())
    n_escluse_nan = len(pool) - len(pool_candidata)
    if n_escluse_nan > 0:   #indicazioni su quante istituzioni sono state escluse per troppi valori mancanti (in un qualsiasi split)
        print(f"nan-threshold={config['nan-threshold']}: {n_escluse_nan} istituzioni escluse per troppi valori mancanti")

    # Nessuna divisione per client: le finestre di ogni istituzione finiscono tutte nello stesso
    # mucchio, split per split.
    finestre = {"train": [], "validation": [], "test": []}
    pool_valido = []  # istituzioni finite davvero nel pool: va stampato, e' l'insieme su cui vale il confronto
    n_troppo_piccole = 0  # istituzioni scartate perche' senza finestre in almeno uno split
    for institution_id in pool_candidata:
        # carica le finestre dei tre split
        X_train, Y_train = concatena_finestre(dataset.get_train_dataloader(ts_id=institution_id))
        X_val, Y_val = concatena_finestre(dataset.get_val_dataloader(ts_id=institution_id))
        X_test, Y_test = concatena_finestre(dataset.get_test_dataloader(ts_id=institution_id))
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            # scarta se un split è vuoto
            n_troppo_piccole += 1
            continue
        # accumula le finestre valide
        finestre["train"].append((X_train, Y_train))
        finestre["validation"].append((X_val, Y_val))
        finestre["test"].append((X_test, Y_test))
        pool_valido.append(institution_id)

    if n_troppo_piccole > 0:
        print(f"{n_troppo_piccole} istituzioni escluse perche' senza finestre in almeno uno split")
    if not pool_valido:
        raise RuntimeError("Nessuna istituzione valida: controlla data-root e nan-threshold.")

    print(f"Istituzioni usate (tutte insieme, nessuna divisione per client): {len(pool_valido)}")

    # concatena le finestre di tutte le istituzioni
    concatenati = []
    for split in ("train", "validation", "test"):
        X = np.concatenate([X for X, _ in finestre[split]], axis=0)
        Y = np.concatenate([Y for _, Y in finestre[split]], axis=0)
        concatenati.append((X, Y))
    return concatenati


def metriche(trues, preds):
    """mse, rmse, r2, mae fra predizioni e valori reali."""
    errori = preds - trues
    mse = errori.square().mean().item()
    mae = errori.abs().mean().item()
    sst = (trues - trues.mean()).square().sum().item()  # varianza totale dei valori reali
    r2 = 1.0 - errori.square().sum().item() / sst if sst > 0 else 0.0
    return mse, mse ** 0.5, r2, mae


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Un passaggio completo sui dati di training, con un aggiornamento dei pesi.

    Identica a quella del non-IID SENZA il termine prossimale di FedProx (mu/global_params):
    qui non c'e' nessun modello globale da cui restare vicini, quindi la loss e' la MSE pura.
    Tutto il resto (ordine delle operazioni, zero_grad/backward/step, metrica riportata) e'
    invariato.
    """
    model.train()
    mse_pure = []  # MSE di ogni mini-batch

    for X, Y in loader:  # loader dà già mini-batch pronti (vedi dividi_in_batch)
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)[:, :, TARGET_FEATURE_INDEX]

        optimizer.zero_grad()       # azzero i gradienti accumulati dal mini-batch precedente
        preds = model(X)            # forward pass sul mini-batch
        mse = criterion(preds, Y)   # MSE tra predizioni e valori reali

        mse.backward()              # calcola i gradienti
        optimizer.step()            # applica i gradienti, aggiorna i pesi

        mse_pure.append(mse.item())

    return torch.tensor(mse_pure).mean().item()  # MSE media di tutti i mini-batch dell'epoca


def test(model, loader, device):
    """Predizioni del modello su un loader, senza allenarlo.

    Restituisce (trues, preds) come tensori (n_finestre, prediction_window). Le metriche
    le calcola il chiamante con metriche().
    """
    model.to(device)
    model.eval()
    preds, trues = [], []

    with torch.no_grad():  # non calcolo i gradienti, inutile durante la valutazione
        for X, Y in loader:  # loader dà già mini-batch pronti (vedi dividi_in_batch)
            X = torch.from_numpy(X).float().to(device)
            # (batch, prediction_window): tutti gli step, non solo il primo
            Y = torch.from_numpy(Y).float().to(device)[:, :, TARGET_FEATURE_INDEX]
            preds.append(model(X))
            trues.append(Y)

    return torch.cat(trues), torch.cat(preds)
