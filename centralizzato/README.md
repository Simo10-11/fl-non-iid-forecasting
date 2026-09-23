# Forecasting centralizzato del traffico di rete (CESNET-TimeSeries24)

Previsione del traffico di rete sul dataset [CESNET-TimeSeries24](https://github.com/koumajos/CESNET-TimeSeries24), con due approcci fianco a fianco:

- **`main.py`** — un **unico modello centralizzato** (LSTM bidirezionale univariata su `n_bytes`), addestrato su PyTorch con tutti i dati di tutte le istituzioni in un solo pool. Nessuna federazione: niente client, niente round, niente Flower.
- **`baseline.py`** — tre **previsioni banali senza nessun modello e nessun addestramento**, calcolate sullo stesso identico test set del centralizzato. Servono da riferimento minimo: se l'LSTM non le batte tutte, il problema non è il tuning degli iperparametri ma qualcosa a monte.

Questo progetto è il **termine di paragone** dei due progetti federati ([`fl-iid-forecasting`](https://github.com/Simo10-11/fl-iid-forecasting), [`fl-non-iid-forecasting`](https://github.com/Simo10-11/fl-non-iid-forecasting)): rappresenta il caso ideale in cui i dati potessero davvero essere raccolti in un unico posto. Modello, split cronologico per istituzione, scaler MinMax fittato solo sul train, finestre di 168 ore per predirne 24 e iperparametri sono **identici** a quelli dei federati, così l'unica differenza misurata è la federazione in sé.

---

## Requisiti

- **Python 3.11+** (serve `tomllib` della libreria standard per leggere `config.toml`)
- Dipendenze: `torch`, `cesnet-tszoo`, `numpy`, `scikit-learn` (quest'ultima usata solo da `baseline.py`), `ipython` (dipendenza transitiva di `cesnet-tszoo`, non installata automaticamente da tutti i resolver: senza va in `ModuleNotFoundError: No module named 'IPython'`)
- ~150 MB liberi su disco: alla prima esecuzione `cesnet-tszoo` scarica automaticamente il dataset dentro `data/`

---

## Installazione

```bash
git clone https://github.com/Simo10-11/forecasting-centralizzato.git
cd forecasting-centralizzato

# crea e attiva un virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch cesnet-tszoo numpy scikit-learn ipython
```

---

## Esecuzione

Non serve nessun argomento in nessuno dei due script: tutta la configurazione sta in `config.toml`, ed entrambi la leggono allo stesso modo (`baseline.py` riusa proprio le funzioni di caricamento di `task.py`, quindi vedono sempre lo stesso dataset). Alla prima esecuzione (di uno qualsiasi dei due) il dataset viene scaricato in `data/`, che viene creata da sola; dalle volte successive si parte subito.

Il percorso del dataset è risolto **rispetto alla cartella del progetto**, non a quella da cui lanci il comando: funziona identico da qualunque directory e su qualunque macchina, senza percorsi assoluti da sistemare dopo un `git clone`.

### Modello centralizzato

```bash
python main.py
```

Allena l'LSTM con early stopping e model selection sulla validation (vedi [Output atteso](#output-atteso)), poi valuta il checkpoint migliore sul test set. Con `save-model = true` (default) salva i pesi in `final_model.pt`.

### Baseline

```bash
python baseline.py
```

Vedi la sezione [Baseline](#baseline-1) qui sotto per cosa calcola e perché.

---

## Baseline

`baseline.py` non allena nulla: per ogni finestra di test copia dei valori già osservati in `X` (le 168 ore di input) e li confronta con `Y` (le 24 ore vere da predire), sulle **stesse identiche finestre** viste dal modello centralizzato. Tre varianti, dalla più ingenua alla più informata:

| Previsione | Cosa copia | Intuizione |
|---|---|---|
| `ora precedente` | l'ultima ora osservata, ripetuta 24 volte | "il traffico tra un'ora sarà uguale ad adesso" |
| `stessa ora, 24h prima` | le ultime 24 ore osservate | "il traffico di domani sarà come quello di oggi" (stagionalità giornaliera) |
| `stessa ora, 168h prima (inizio finestra)` | le prime 24 ore della finestra di input | "il traffico di domani sarà come quello di una settimana fa" (stagionalità settimanale) |

---

## Output atteso

**`main.py`** stampa, in ordine:

- quante istituzioni sono state escluse dal `nan-threshold` e quante ne restano nel pool (è l'insieme su cui vale il confronto con i federati)
- il numero di finestre di train / validation / test
- una riga per epoca con MSE di training e di validation, marcata `<- nuovo migliore` quando la validation migliora
- l'epoca del checkpoint migliore e, una sola volta a fine training, le metriche di **test** (`mse`, `rmse`, `r2`, `mae`)

Il test set non viene mai toccato durante il training: viene valutato una volta sola, alla fine, sul checkpoint migliore.

**`baseline.py`** stampa il numero di istituzioni e di finestre di test, seguiti da una riga di metriche per ciascuna delle tre previsioni descritte sopra — comparabili direttamente con la riga di test finale di `main.py`.

---

## Struttura del progetto

```
forecasting-centralizzato
├── main.py          # ciclo di training, early stopping, model selection e test finale
├── task.py          # modello LSTM, caricamento/preparazione dati, training di un'epoca, metriche
├── baseline.py      # previsioni banali senza addestramento, sullo stesso test set
├── config.toml      # tutta la configurazione (dataset, split, modello, training)
├── data/            # dataset scaricato automaticamente (in .gitignore, non versionato)
└── README.md
```

---

## Confronto con i progetti federati

Perché il confronto sia onesto, centralizzato e federato devono lavorare sugli **stessi dati** e partire dalle **stesse condizioni**. Un punto a cui fare attenzione prima di mettere i numeri in tabella:

- **Pool di istituzioni.** Qui il training usa *sempre tutte* le istituzioni valide. Nei federati è `--num-supernodes` a decidere quante istituzioni entrano nella run (un client = un'istituzione), scegliendone un sottoinsieme casuale. Se il run federato ne usa 10 e questo ne usa 271, i due test set sono diversi e le metriche non sono confrontabili: fai girare il federato con `num-supernodes` pari al numero di istituzioni valide, oppure limita questo progetto allo stesso sottoinsieme.
