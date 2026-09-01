# Federated Forecasting del traffico di rete (CESNET-TimeSeries24)

Previsione del traffico di rete con **federated learning**, usando [Flower](https://flower.ai) e PyTorch sul dataset [CESNET-TimeSeries24](https://github.com/koumajos/CESNET-TimeSeries24). Distribuzione **non-IID**: ogni client rappresenta un'unica istituzione reale del dataset, e riceve solo i dati di quella istituzione (mai un mix con altre). Ogni istituzione ha uno split temporale train/test (70/30, cronologico): il 70% (train) va per intero al client di quella istituzione; il 30% (test) viene diviso, istituzione per istituzione, tra un test set **globale** del server (mai visto dai client, rappresentativo di tutto il traffico) e un test **locale** per il client di quella istituzione, usato per la federated evaluation.

---

## Requisiti

- Python 3.10+
- Dipendenze principali (vedi `pyproject.toml`): `flwr[simulation]`, `torch`, `cesnet-tszoo`, `scikit-learn`, `numpy`
- ~150 MB liberi su disco: alla prima esecuzione `cesnet-tszoo` scarica automaticamente il dataset

---

## Installazione

```bash
git clone <url-del-repo>
cd fl-non-iid-forecasting

# crea e attiva un virtual environment
python -m venv flwr-env
source flwr-env/bin/activate      # Windows: flwr-env\Scripts\activate

# installa il progetto e le sue dipendenze
pip install -e .
```

---

## Esecuzione

Avvia la simulazione federata con i parametri di default:

```bash
flwr run . --stream
```

`--stream` mostra i log in tempo reale.

Il numero di client simulati e le risorse per client si passano a parte, come `--federation-config`. In questo progetto un client = un'istituzione, ed è `num-supernodes` stesso a decidere quante istituzioni usare: se ne sceglie un sottoinsieme casuale ma riproducibile (stesso `random-state` → stesso sottoinsieme) tra quelle valide per `nan-threshold`:

```bash
flwr run . --stream --federation-config 'num-supernodes=10 client-resources-num-cpus=6'
```

Se chiedi più supernodi delle istituzioni disponibili nel dataset, l'esecuzione si interrompe subito con un errore che indica il massimo utilizzabile.

Per rendere permanente questa configurazione (invece di ripassarla a ogni run):

```bash
flwr federation simulation-config @none/default local-simulation \
    --num-supernodes 10 \
    --client-resources-num-cpus 6 \
    --client-resources-num-gpus 0.5
```

---

## Configurazione

Tutti gli iperparametri stanno in `pyproject.toml`, sotto `[tool.flwr.app.config]`:

- **Dataset**: feature target, dimensione delle finestre di input/predizione, seed
- **Split dati**: `train-time-period`/`test-time-period` (70/30 cronologico per istituzione: il 70% train va tutto al client di quella istituzione, il 30% test viene diviso per istituzione tra server e client), `global-test-fraction` (per ciascuna istituzione, frazione delle sue finestre di test riservata al server come test set globale, isolata prima di lasciare il resto come test locale del client)
- **Modello**: dimensioni della LSTM, learning rate, batch size
- **Federated learning**: numero di round, epoche locali per round, frazione di client per training/valutazione, minimo di client richiesti, salvataggio del modello finale

Per un run singolo si possono sovrascrivere senza modificare il file:

```bash
flwr run . --stream --run-config 'num-server-rounds=20 learning-rate=0.005'
```

---

## Output atteso

Durante l'esecuzione vengono stampati:

- il numero di istituzioni effettivamente usate in questa run (= `--num-supernodes`) e la dimensione del test set globale del server, una sola volta a inizio run
- un log per ogni fase di ogni round (`train -> N client coinvolti`, `evaluate -> N client coinvolti`)
- le metriche aggregate di training (`train_mse`) e di valutazione federata (`mse`, `rmse`, `r2`, `mae`) sui client
- le metriche di **global evaluation** (`mse`, `rmse`, `r2`, `mae`) calcolate dal server sul proprio test set globale, prima del round 1 e dopo ogni round

Se `save-model = true` (default), a fine run il modello globale finale viene salvato come `final_model.pt` nella cartella del progetto.

---

## Struttura del progetto

```
fl-non-iid-forecasting
├── fl_noniid_netforecast
│   ├── task.py          # modello LSTM, caricamento/preparazione dati, training e metriche
│   ├── client_app.py    # ClientApp: training e valutazione locale di ogni client
│   └── server_app.py    # ServerApp: strategia FedAvg e ciclo dei round
├── pyproject.toml       # dipendenze e configurazione (dataset, modello, federated learning)
└── README.md
```
