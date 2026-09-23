# Forecasting del traffico di rete (CESNET-TimeSeries24)

Previsione del traffico di rete (LSTM + PyTorch) sul dataset [CESNET-TimeSeries24](https://github.com/koumajos/CESNET-TimeSeries24), confrontando **federated learning** (Flower) con un **termine di paragone centralizzato**. Questa repo raccoglie i tre esperimenti correlati, ciascuno nella propria cartella autosufficiente (propria configurazione, `README.md`):

- **[`iid/`](iid/README.md)** — federato, distribuzione IID: il train set di tutte le istituzioni viene mescolato e ripartito in fette uguali tra i client.
- **[`non-iid/`](non-iid/README.md)** — federato, distribuzione non-IID: ogni client rappresenta un'unica istituzione reale (nessun mix), con aggregazione FedProx.
- **[`centralizzato/`](centralizzato/README.md)** — nessuna federazione: un unico modello addestrato su tutti i dati in un solo pool, più una baseline senza training. Serve da termine di paragone per gli altri due.

Per dettagli su modello, split dei dati, configurazione e comandi di esecuzione, vedi il README dentro ciascuna cartella.

## Esecuzione

`iid/` e `non-iid/` sono progetti Flower e vanno eseguiti dalla propria cartella (Flower legge `pyproject.toml` dalla working directory corrente):

```bash
cd iid       # oppure: cd non-iid
python -m venv flwr-env && source flwr-env/bin/activate
pip install -e .
flwr run . --stream
```

`centralizzato/` è uno script standalone (nessun Flower):

```bash
cd centralizzato
python -m venv .venv && source .venv/bin/activate
pip install torch cesnet-tszoo numpy scikit-learn ipython
python main.py
```
