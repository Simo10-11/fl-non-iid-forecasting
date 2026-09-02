"""Federated Learning su CESNET-TimeSeries24"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import AgreggationType, SourceType

TARGET_FEATURE_INDEX = 0    # uso solo la feature target, quindi l'indice è 0, (modello univariato)


def get_device():
    """Device di calcolo: GPU se disponibile, altrimenti CPU"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMForecast(nn.Module):
    """LSTM per forecasting: prende in input una finestra di training e predice gli step della finestra di predizione"""

    # I valori qui sotto sono solo DEFAULT della firma
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, output_size=1):  # costruisce l'architettura della rete
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


#run_config pesco la configurazione di run dal server, che la passa a tutti i client ( vedi [tool.flwr.app.config] in pyproject.toml)
def build_model(run_config:dict):
    """Costruisce SOLO l'architettura, senza allenarla.
    Legge gli iperparametri dal run_config e restituisce una LSTM nuova,
    con pesi casuali. 
    Seed viene fissato una volta sola dal server (i client non inizializzano mai riceveranno i pesi globali)
    """
    return LSTMForecast(
        input_size=1,  # modello univariato: una sola feature (target-feature)
        hidden_size=int(run_config["hidden-size"]),
        num_layers=int(run_config["num-layers"]),
        dropout=float(run_config["dropout"]),
        output_size=int(run_config["prediction-window-size"]),
    )


def apri_dataset(run_config:dict):
    """Apre il dataset CESNET-TimeSeries24.
    Non carica ancora i dati: serve poi 
    Sta in una funzione perché la chiameranno in altre funzioni (istituzioni_disponibili e load_data).
    """
    return CESNET_TimeSeries24.get_dataset(
        data_root=str(run_config["data-root"]),
        source_type=SourceType.INSTITUTIONS,
        aggregation=AgreggationType.AGG_1_HOUR,
        dataset_type="time_based",
        display_details=False,  # niente output verboso: qui i client sono centinaia
    )


def istituzioni_disponibili(run_config: dict):
    """Id reali di TUTTE le istituzioni disponibili nel dataset, PRIMA di ogni filtro: serve
    solo per contare quante ne vengono scartate dal nan-threshold"""
    ids = apri_dataset(run_config).get_available_ts_indices()["id_institution"]
    return ids.tolist()


def concatena_finestre(loader): #lo uso ogni volta per ogni isituzione, trasmfromo il lodaer di censt_tzoo in semplici array
    """Scorre un DataLoader di cesnet_tszoo (una finestra alla volta) e concatena
    tutte le finestre in due unici array: (n_finestre, window_size, 1)."""
    X_all, Y_all = [], []
    for X, Y in loader: #il loader mi da una finestra alla volta, le inglobo rispettivamente in un contenitore X e Y
        X_all.append(X)
        Y_all.append(Y)
    return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0) #concateno tutte le finestre in un unico array numpy (n_finestre, window_size, 1)


_istituzioni_e_finestre_cache: tuple | None = None  # cache in-memory (pool_valido, train, test), costruita una sola volta per run


def prepara_istituzioni_finestre(num_partitions: int, run_config: dict):
    """Apre il dataset, lo configura, e restituisce le num_partitions istituzioni assegnate ai client di questa run, e
    le loro finestre di train/test, TENUTE SEPARATE istituzione per istituzione

    num_partitions (cioè --num-supernodes) decide DIRETTAMENTE quante istituzioni usare: un
    client = un'istituzione

    Lo scaler MinMax è fittato dalla libreria SOLO sul periodo "train", quindi non vede mai
    dati da predire.

    Cachato per l'intera run (train e test condividono la stessa TimeBasedConfig, quindi vanno
    costruiti insieme una sola volta, invece che una volta per periodo come nella versione IID).
    """
    global _istituzioni_e_finestre_cache
    if _istituzioni_e_finestre_cache is not None:
        return _istituzioni_e_finestre_cache

    pool = istituzioni_disponibili(run_config)

    dataset = apri_dataset(run_config)
    config = TimeBasedConfig(
        ts_ids=pool,    #lista di più isitituzioni
        train_time_period=float(run_config["train-time-period"]),
        test_time_period=float(run_config["test-time-period"]),
        features_to_take=[str(run_config["target-feature"])],
        sliding_window_size=int(run_config["training-window-size"]),
        sliding_window_prediction_size=int(run_config["prediction-window-size"]),
        sliding_window_step=int(run_config["prediction-window-size"]),
        random_state=int(run_config["random-state"]),
        transform_with="min_max_scaler",    # scaler viene fittato solo sul training set
        nan_threshold=float(run_config["nan-threshold"]),  # esclude istituzioni con troppi NaN (vedi docstring sopra)
        fill_missing_with="forward_filler", # riempie i NaN con l'ultimo valore disponibile, 0 se primo valore della serie
        include_ts_id=False,
        include_time=False,
    )
    dataset.set_dataset_config_and_initialize(config, display_config_details=None)
    pool_nan_valido = sorted(dataset.dataset_config.ts_ids.tolist())  # ordine fisso: base del mapping partition_id -> institution_id
    n_escluse_nan = len(pool) - len(pool_nan_valido)
    if n_escluse_nan > 0:   #indicazioni su quante istituzioni sono state escluse per troppi valori mancanti
        print(f"nan-threshold={run_config['nan-threshold']}: {n_escluse_nan} istituzioni escluse per troppi valori mancanti")

    if num_partitions > len(pool_nan_valido):   #se numero client > numero isituzioni disponibili (che rispttano il Nan), lancio errore
        raise ValueError(
            f"Il dataset ha solo {len(pool_nan_valido)} istituzioni valide (dopo nan-threshold), "
            f"ma la federazione è stata avviata con {num_partitions} supernodi (client). In "
            f"questo progetto un client = un'istituzione, "
            f"{len(pool_nan_valido)}: riduci --num-supernodes a un valore <= {len(pool_nan_valido)}."
        )
    if num_partitions < len(pool_nan_valido):   #se numero client < numero isituzioni disponibili (che rispttano il Nan), scelgo un sottoinsieme casuale di istituzioni
        rng = np.random.default_rng(int(run_config["random-state"]))    #casuale
        pool_candidata = sorted(rng.choice(pool_nan_valido, size=num_partitions, replace=False).tolist())
        print(f"--num-supernodes={num_partitions}: sottoinsieme scelto casualmente (seed={run_config['random-state']}) tra le {len(pool_nan_valido)} istituzioni valide per nan-threshold")
    else:   #numero client = numero isituzioni disponibili, uso tutte le istituzioni
        pool_candidata = pool_nan_valido  
      
    # per ogni istituzione, concateno tutte le finestre di train e test in due array (X, Y) separati
    train_per_istituzione, test_per_istituzione = {}, {}  # finestre per istituzione
    pool_valido = []  # istituzioni che risulteranno effettivamente usabili
    n_troppo_piccole = 0  # conta le istituzioni scartate per pochi dati (non ce ne saranno, hanno tutte gli stessi dati)
    for institution_id in pool_candidata:  # scorre le istituzioni scelte per questa run
        X_train, Y_train = concatena_finestre(dataset.get_train_dataloader(ts_id=institution_id))  # carica le finestre di train
        X_test, Y_test = concatena_finestre(dataset.get_test_dataloader(ts_id=institution_id))  # carica le finestre di test
        if len(X_train) == 0 or len(X_test) == 0:  # istituzione senza abbastanza dati
            n_troppo_piccole += 1  # segna un'istituzione scartata
            continue  # non la aggiunge al pool
        train_per_istituzione[institution_id] = (X_train, Y_train)  # salva le finestre di train
        test_per_istituzione[institution_id] = (X_test, Y_test)  # salva le finestre di test
        pool_valido.append(institution_id)  # istituzione confermata valida


    print(f"Istituzioni usate in questa run (= numero di client): {len(pool_valido)}.\n")

    _istituzioni_e_finestre_cache = (pool_valido, train_per_istituzione, test_per_istituzione)
    return _istituzioni_e_finestre_cache


def dividi_in_batch(X, Y, batch_size):
    """Taglia X e Y in blocchi consecutivi da batch_size finestre (l'ultimo può essere più corto).
    Restituisce la lista di questi blocchi come coppie (X_batch, Y_batch)."""
    return [(X[i:i + batch_size], Y[i:i + batch_size]) for i in range(0, len(X), batch_size)]


_split_test_cache: tuple | None = None  # cache in-memory di split_test, costruita una sola volta


def split_test(num_partitions: int, run_config: dict):
    """Split non-IID a 2 livelli del pool TEST

    Per ciascuna istituzione:
    1. si isola una fetta (global-test-fraction) delle sue finestre di test per il test set
       GLOBALE del server;
    2. il resto resta come test LOCALE di quella sola istituzione, quindi di un solo client

    Cachato per l'intera run: senza cache, ogni client che chiama load_data(split="test")
    rifarebbe da capo lo split di TUTTE le istituzioni solo per leggere la propria fetta
    """
    global _split_test_cache  # dichiara che modifico la cache globale
    if _split_test_cache is not None:  # se già calcolata in questa run
        return _split_test_cache  # riusa il risultato salvato prima

    pool_valido, _, finestre_test = prepara_istituzioni_finestre(num_partitions, run_config)  # istituzioni e finestre di test

    seed = int(run_config["random-state"])  # seed fisso per riproducibilità
    frac_globale = float(run_config["global-test-fraction"])  # frazione riservata al server

    X_globale_parti, Y_globale_parti = [], []  # accumulano le fette globali
    locale_per_istituzione = {}  # fetta locale, una per istituzione
    for institution_id in pool_valido:  # scorre ogni istituzione della run
        X, Y = finestre_test[institution_id]  # finestre di test di questa istituzione
        rng = np.random.default_rng((seed, institution_id))  # seed diverso ma riproducibile per ogni istituzione
        permutazione = rng.permutation(len(X))  # ordine casuale delle finestre

        n_globale = int(len(permutazione) * frac_globale)  # quante finestre vanno al globale
        idx_globale, idx_locale = permutazione[:n_globale], permutazione[n_globale:]  # divide indici globali e locali

        X_globale_parti.append(X[idx_globale])  # aggiunge la fetta X globale
        Y_globale_parti.append(Y[idx_globale])  # aggiunge la fetta Y globale
        locale_per_istituzione[institution_id] = (X[idx_locale], Y[idx_locale])  # salva la fetta locale

    X_globale = np.concatenate(X_globale_parti)  # unisce le fette X di tutti
    Y_globale = np.concatenate(Y_globale_parti)  # unisce le fette Y di tutti
    _split_test_cache = (X_globale, Y_globale, locale_per_istituzione)  # salva il risultato in cache
    return _split_test_cache  # restituisce il risultato appena calcolato


_test_globale_cache: list | None = None  # cache in-memory del test set globale del server, costruito una sola volta


def load_test_globale(num_partitions: int, run_config: dict):
    """Carica il test set globale del server: l'unione, su tutte le istituzioni, della fetta
    isolata da ciascuna prima di lasciare il resto come test locale del client corrispondente."""
    global _test_globale_cache
    if _test_globale_cache is not None:
        return _test_globale_cache

    X_globale, Y_globale, _ = split_test(num_partitions, run_config)

    batch_size = int(run_config["batch-size"])
    _test_globale_cache = dividi_in_batch(X_globale, Y_globale, batch_size)
    return _test_globale_cache


_load_data_cache: dict[tuple[int, int, str], list] = {}  # cache in-memory dei batch già costruiti, per client e split


def load_data(partition_id: int, num_partitions: int, run_config: dict, split: str):
    """
    Dato l'indice di un client, costruisce il suo dataset non-IID per lo split richiesto:
    un client = un'istituzione , quindi riceve SOLO dati della propria istituzione

    - "train": TUTTE le finestre di training della sua istituzione 
    - "test": la fetta di test LOCALE della sua istituzione, cioè quello che resta dopo aver
      isolato la parte destinata al test set globale del server (vedi split_test).

    Il risultato viene cachato: si esegue una sola volta per client per l'intera durata della
    run, invece che una volta per round.
    """
    cache_key = (partition_id, num_partitions, split)
    if cache_key in _load_data_cache:
        return _load_data_cache[cache_key]  # hit: nei round successivi salta subito il ricaricamento da cesnet_tszoo

    # prepara_istituzioni_finestre costruisce il mapping partition_id (identificstivo client)-> institution_id e carica le finestre di train/test
    pool_valido, train_per_istituzione, _ = prepara_istituzioni_finestre(num_partitions, run_config)
    institution_id = pool_valido[partition_id]

    if split == "train":
        X, Y = train_per_istituzione[institution_id]
    else:   #test
        _, _, locale_per_istituzione = split_test(num_partitions, run_config)
        X, Y = locale_per_istituzione[institution_id]

    batch_size = int(run_config["batch-size"])
    result = dividi_in_batch(X, Y, batch_size)
    _load_data_cache[cache_key] = result  # miss: memorizza il risultato, così viene calcolato una volta sola per client
    return result




def statistiche_additive(trues, preds):
    """Statistiche additive (somme, non medie) calcolate da un client sul proprio test set locale.

    Servono al server per ricostruire mse/rmse/r2/mae esatti: rmse, r2 e mae non sono lineari,
    quindi mediarli, anche pesando per numero di finestre, non equivale a calcolarli sui dati
    concatenati. Nessun dato grezzo lascia il client: solo 5 numeri.
     (esempio: due client con rmse locale 1 e 3 sullo stesso numero di finestre non danno un
    rmse globale di 2, ma sqrt(5) ≈ 2.236).
    """
    errors = preds - trues
    sse = errors.square().sum().item()
    sum_abs_error = errors.abs().sum().item()
    sum_y = trues.sum().item()
    sum_y_sq = trues.square().sum().item()
    num_values = trues.numel()
    return sse, sum_abs_error, sum_y, sum_y_sq, num_values


def metriche_da_statistiche_additive(sse, sum_abs_error, sum_y, sum_y_sq, num_values):
    """Ricalcola mse/rmse/r2/mae a partire da statistiche additive sommate su più client"""
    mse = sse / num_values
    rmse = mse ** 0.5
    mae = sum_abs_error / num_values
    sst = sum_y_sq - (sum_y ** 2 / num_values)  # somma dei quadrati degli scarti dalla media GLOBALE dei target
    if sst > 0:
        r2 = 1.0 - sse / sst
    else:
        r2 = 1.0 if sse == 0 else 0.0  # varianza nulla: predizione perfetta o convenzionalmente 0
    return mse, rmse, r2, mae




def train_one_epoch(model, loader, criterion, optimizer, device, mu, global_params):
    """Un passaggio completo sui dati di training locali, con un aggiornamento dei pesi.

    mu, global_params: termine prossimale di FedProx. Flower gestisce
    FedProx solo lato server (manda mu ai client dentro il ConfigRecord): il termine va
    aggiunto qui alla loss, lato client, perché è un vincolo sul TRAINING locale, non
    qualcosa che il server possa calcolare guardando solo i pesi finali restituiti.
    """
    model.train()
    mse_pure = []  # MSE pura di ogni mini-batch (SENZA il termine prossimale): quella riportata come train_mse, cosi' resta confrontabile anche quando mu>0

    for X, Y in loader:  # loader dà già mini-batch pronti (vedi load_data)
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)[:, :, TARGET_FEATURE_INDEX]

        optimizer.zero_grad()       # azzero i gradienti accumulati dal mini-batch precedente
        preds = model(X)            # forward pass sul mini-batch
        mse = criterion(preds, Y)   # MSE tra predizioni e valori reali
        loss = mse                  # loss che va a backward: MSE, + eventuale termine prossimale sotto
        
        if mu > 0:  # se FedProx attivo (mu=0 salta tutto)
            # Penalizza quanto il modello locale si e' allontanato dai pesi GLOBALI con cui
            # e' partito questo round: contiene il "client drift" atteso quando i client
            # sono molto eterogenei (qui: un client = un'istituzione diversa).
            termine_prossimale = sum(
                (w_locale - w_globale).pow(2).sum()      # distanza al quadrato, per parametro
                for w_locale, w_globale in zip(model.parameters(), global_params)  # confronta locale con globale
            )
            loss = loss + (mu / 2) * termine_prossimale  # aggiunge la penalita' pesata alla loss

        loss.backward()             # calcola i gradienti della loss combinata
        optimizer.step()            # applica i gradienti, aggiorna i pesi

        mse_pure.append(mse.item())  # salva la MSE pura, non la loss combinata

    return torch.tensor(mse_pure).mean().item()  # MSE pura media di tutti i mini-batch dell'epoca


def train(model, train_loader, epochs, lr, device, mu=0.0):
    """Il client allena il modello locale per epoche partendo dai pesi ricevuti dal server.

    mu=0.0 (default): nessun termine prossimale, comportamento identico a prima di FedProx.
    """
    model.to(device)

    # Copia dei pesi GLOBALI di questo round, presa dopo aver spostato il modello sulla device
    # (cosi' e' garantito che stia sulla stessa device dei pesi locali). Serve solo come
    # riferimento per il termine prossimale: non partecipa mai a backward/optimizer.step().
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    criterion = nn.MSELoss()  # loss per regressione: minimizzo la MSE tra predizioni e valori reali (seguo il paper, anche se sensibile agli outlier)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # lr fisso che arriva dal server

    avg_loss = 0.0      # loss media dell'epoca corrente, che viene aggiornata a ogni epoca e restituita alla fine
    for _ in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, mu, global_params)

    return avg_loss  # loss media dell'ULTIMA epoca locale


def test(model, loader, device):
    """Valuta il modello sul test set locale senza allenarlo (chiamata dal client). """
    model.to(device)
    model.eval()
    preds, trues = [], []

    with torch.no_grad():  # non calcolo i gradienti, inutile durante la valutazione
        for X, Y in loader:  # loader dà già mini-batch pronti (vedi load_data)
            X = torch.from_numpy(X).float().to(device)
            # (batch, prediction_window): tutti gli step, non solo il primo
            Y = torch.from_numpy(Y).float().to(device)[:, :, TARGET_FEATURE_INDEX]
            preds.append(model(X))
            trues.append(Y)

    # concateno tutte le predizioni e i valori reali in un unico tensore (N, prediction_window)
    preds = torch.cat(preds)
    trues = torch.cat(trues)

    # Appiattisco le matrici (n_finestre, prediction_window) in un unico vettore, cosi' le
    # metriche sono calcolate su tute le predizioni insieme. 
    trues_np = trues.cpu().numpy().flatten()
    preds_np = preds.cpu().numpy().flatten()

    # metriche in scala normalizzata
    mse = F.mse_loss(preds, trues)
    rmse = root_mean_squared_error(trues_np, preds_np)
    r2 = r2_score(trues_np, preds_np)
    mae = mean_absolute_error(trues_np, preds_np)

    return mse.item(), float(rmse), float(r2), float(mae), trues, preds
