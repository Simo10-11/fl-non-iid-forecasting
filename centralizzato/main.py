""" CENTRALIZZATA: tutti i dati di tutte le istituzioni in un unico posto, un solo
modello, nessuna federazione. Si lancia con `python main.py`, non serve Flower.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from task import (
    build_model,
    carica_config,
    dividi_in_batch,
    get_device,
    metriche,
    prepara_dataset_globale,
    test,
    train_one_epoch,
    CARTELLA,
)


def valuta(model, loader, device):
    """mse/rmse/r2/mae del modello su un loader."""
    trues, preds = test(model, loader, device)
    return metriche(trues, preds)


def main():
    config = carica_config()

    # Seed fissato una volta sola, prima di costruire il modello: rende riproducibile
    # l'inizializzazione dei pesi, esattamente come fa il server nei federati.
    seed = int(config["random-state"])
    torch.manual_seed(seed)
    device = get_device()

    # Split cronologico per istituzione, poi tutte le finestre concatenate in un unico
    # dataset per split (vedi prepara_dataset_globale).
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = prepara_dataset_globale(config)
    print(f"Finestre totali -> train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}\n")

    batch_size = int(config["batch-size"])
    val_loader = dividi_in_batch(X_val, Y_val, batch_size)
    # Il test set viene valutato come UN UNICO BLOCCO: test() accumula tutte le predizioni e le
    # concatena PRIMA di calcolare le metriche
    test_loader = dividi_in_batch(X_test, Y_test, batch_size)

    model = build_model(config).to(device)
    criterion = nn.MSELoss()  # stessa loss dei federati: MSE
    # L'optimizer si crea UNA volta sola e vive per tutto il training: nei federati invece si
    # ricrea a ogni round, perche' ogni round e' un training locale che riparte da zero.
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning-rate"]))

    max_epochs = int(config["max-epochs"])
    patience = int(config["early-stopping-patience"])
    rng = np.random.default_rng(seed)  # usato solo per l'ordine delle finestre

    # Stesso criterio di model selection dei federati: si tiene il checkpoint alla validation mse
    # piu' bassa vista finora, non l'ultimo.
    migliore = {"mse": float("inf"), "state": None, "epoca": None}
    epoche_senza_miglioramento = 0

    for epoca in range(1, max_epochs + 1):
        # Ordine mescolato a ogni epoca: le finestre di tutte le istituzioni sono concatenate,
        # quindi senza mescolare ogni mini-batch conterrebbe finestre di una sola istituzione
        # e i gradienti sarebbero fortemente correlati.
        ordine = rng.permutation(len(X_train))
        train_loader = dividi_in_batch(X_train[ordine], Y_train[ordine], batch_size)

        train_mse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, _, _, _ = valuta(model, val_loader, device)

        if val_mse < migliore["mse"]:
            migliore["mse"] = val_mse
            migliore["epoca"] = epoca
            # Copia indipendente dei pesi: il modello continua ad allenarsi dopo questo punto.
            migliore["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}        # Copia dei pesi
            epoche_senza_miglioramento = 0
            print(f"epoca {epoca:3d} | train mse={train_mse:.5f} | validation mse={val_mse:.5f}  <- nuovo migliore")
        else:
            epoche_senza_miglioramento += 1
            print(f"epoca {epoca:3d} | train mse={train_mse:.5f} | validation mse={val_mse:.5f}")
            if epoche_senza_miglioramento >= patience:
                print(f"\nEarly stopping: {patience} epoche consecutive senza miglioramento della validation mse.")
                break

    # Model selection: si riparte dal checkpoint migliore, non dai pesi dell'ultima epoca.
    model.load_state_dict(migliore["state"])
    print(f"\nModello migliore: epoca {migliore['epoca']} (mse validazione={migliore['mse']:.5f})")

    # TEST FINALE, una sola volta: fino a qui il test set non e' mai stato toccato.
    mse, rmse, r2, mae = valuta(model, test_loader, device)
    print(
        f"\nTest finale (sul modello migliore, epoca {migliore['epoca']}): "
        f"mse={mse:.5f} rmse={rmse:.4f} r2={r2:.4f} mae={mae:.4f}"
    )

    if config["save-model"]:
        percorso = CARTELLA / "final_model.pt"
        print(f"\nSalvataggio del modello migliore su {percorso}...")
        torch.save(migliore["state"], percorso)


if __name__ == "__main__":
    main()
