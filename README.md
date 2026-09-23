# Federated Forecasting del traffico di rete (CESNET-TimeSeries24)

Previsione del traffico di rete con **federated learning** (Flower + PyTorch) sul dataset [CESNET-TimeSeries24](https://github.com/koumajos/CESNET-TimeSeries24). Questa repo raccoglie i due esperimenti correlati, ciascuno nella propria cartella autosufficiente (proprio `pyproject.toml`, `LICENSE`, `README.md`):

- **[`iid/`](iid/README.md)** — distribuzione IID: il train set di tutte le istituzioni viene mescolato e ripartito in fette uguali tra i client.
- **[`non-iid/`](non-iid/README.md)** — distribuzione non-IID: ogni client rappresenta un'unica istituzione reale (nessun mix), con aggregazione FedProx.

Per dettagli su modello, split dei dati, configurazione e comandi di esecuzione, vedi il README dentro ciascuna cartella.

## Esecuzione

Ogni esperimento va eseguito dalla propria cartella (il progetto Flower legge `pyproject.toml` dalla working directory corrente):

```bash
cd iid       # oppure: cd non-iid
python -m venv flwr-env && source flwr-env/bin/activate
pip install -e .
flwr run . --stream
```
