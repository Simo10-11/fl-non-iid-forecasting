# Risultati esperimenti

## Centralizzato

### Configurazione

| Parametro | Valore |
|---|---|
| target-feature | n_bytes |
| training-window-size | 168 |
| prediction-window-size | 24 |
| train-time-period | 0.7 |
| val-time-period | 0.15 |
| test-time-period | 0.15 |
| nan-threshold | 0.1 |
| hidden-size | 100 |
| num-layers | 1 |
| dropout | 0.0 |
| learning-rate | 0.001 |
| batch-size | 16 |
| max-epochs | 100 |
| early-stopping-patience | 10 |
| save-model | true |
| random-state | variabile per run (vedi tabella risultati) |

### Dati

| Voce | Valore |
|---|---|
| Istituzioni disponibili | 283 |
| Istituzioni escluse (nan-threshold = 0.1) | 8 |
| Istituzioni usate | 275 |
| Finestre di train | 51700 |
| Finestre di validation | 9350 |
| Finestre di test | 9350 |

### Baseline non addestrate (test set globale, 9350 finestre, 24 ore predette)

| Baseline | MSE | RMSE | R² | MAE |
|---|---|---|---|---|
| Ora precedente | 0.00663 | 0.0814 | 0.1663 | 0.0352 |
| Stessa ora, 24h prima | 0.00602 | 0.0776 | 0.2425 | 0.0314 |
| Stessa ora, 168h prima (inizio finestra) | 0.00642 | 0.0801 | 0.1925 | 0.0316 |

### Risultati per seed

| Seed | Epoca migliore | Epoche eseguite | MSE validazione | MSE test | RMSE test | R² test | MAE test |
|---|---|---|---|---|---|---|---|
| 42 | 14 | 24 | 0.00427 | 0.00345 | 0.0587 | 0.5660 | 0.0254 |
| 123 | 10 | 20 | 0.00427 | 0.00350 | 0.0592 | 0.5593 | 0.0259 |
| 456 | 14 | 24 | 0.00425 | 0.00347 | 0.0589 | 0.5640 | 0.0263 |
| 789 | 10 | 20 | 0.00427 | 0.00345 | 0.0587 | 0.5663 | 0.0258 |
| 1000 | 14 | 24 | 0.00425 | 0.00345 | 0.0587 | 0.5661 | 0.0257 |

### Riepilogo (media ± deviazione standard, 5 seed)

| Metrica | Media | Dev. standard |
|---|---|---|
| MSE validazione | 0.004262 | 0.000011 |
| MSE test | 0.003464 | 0.000022 |
| RMSE test | 0.05884 | 0.000219 |
| R² test | 0.56434 | 0.00297 |
| MAE test | 0.02582 | 0.000327 |

## Federated IID — run migliore

### Configurazione

| Parametro | Valore |
|---|---|
| data-root | ./data/cesnet_dataset |
| target-feature | n_bytes |
| training-window-size | 168 |
| prediction-window-size | 24 |
| train-time-period | 0.7 |
| validation-time-period | 0.15 |
| test-time-period | 0.15 |
| nan-threshold | 0.1 |
| num-supernodes (client / istituzioni) | 275 |
| hidden-size | 100 |
| num-layers | 1 |
| dropout | 0.0 |
| learning-rate | 0.001 |
| batch-size | 16 |
| num-server-rounds | 30 |
| local-epochs | 3 |
| fraction-train | 0.3 |
| fraction-evaluate | 1 |
| min-available-clients | 2 |
| save-model | true |
| random-state | 456 |

### Risultato

| Seed | Round migliore | MSE | RMSE | R² | MAE |
|---|---|---|---|---|---|
| 456 | 30 | 0.0036 | 0.0600 | 0.5465 | 0.0277 |

## Federated Non-IID

### local-epochs=6, proximal-mu=0.001

#### Configurazione

| Parametro | Valore |
|---|---|
| data-root | ./data/cesnet_dataset |
| target-feature | n_bytes |
| training-window-size | 168 |
| prediction-window-size | 24 |
| train-time-period | 0.7 |
| val-time-period | 0.15 |
| test-time-period | 0.15 |
| nan-threshold | 0.1 |
| num-supernodes (client / istituzioni) | 275 |
| hidden-size | 100 |
| num-layers | 1 |
| dropout | 0.0 |
| learning-rate | 0.001 |
| batch-size | 16 |
| num-server-rounds | 30 |
| local-epochs | 6 |
| fraction-train | 0.3 |
| fraction-evaluate | 1 |
| min-available-clients | 2 |
| proximal-mu | 0.001 |
| save-model | true |
| random-state | variabile per run (vedi tabella risultati) |

#### Risultati per seed

| Seed | Round migliore | MSE | RMSE | R² | MAE |
|---|---|---|---|---|---|
| 42 | 25 | 0.00440 | 0.06631 | 0.44668 | 0.03314 |
| 123 | 29 | 0.00442 | 0.06645 | 0.44448 | 0.03291 |
| 456 | 30 | 0.00437 | 0.06607 | 0.45068 | 0.03382 |
| 789 | 24 | 0.00437 | 0.06608 | 0.45051 | 0.03303 |
| 1000 | 25 | 0.00444 | 0.06666 | 0.44087 | 0.03258 |

#### Riepilogo (media ± deviazione standard, 5 seed)

| Metrica | Media | Dev. standard |
|---|---|---|
| MSE | 0.00440 | 0.00003 |
| RMSE | 0.06632 | 0.00025 |
| R² | 0.44664 | 0.00416 |
| MAE | 0.03309 | 0.00046 |
