"""ServerApp: strategia di aggregazione FL sui modelli LSTM locali dei client (ciascuno allenato sui dati della propria istituzione, non-IID)."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.app.message_type import MessageType
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_noniid_netforecast.task import build_model, metriche_da_statistiche_additive

app = ServerApp()


def aggrega_train(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    """Media pesata (per num-examples) della train_mse locale di ogni client. La MSE è additiva
    (media di errori al quadrato), quindi qui la media pesata generica di Flower
    (aggregate_metricrecords) è già esatta: non serve una logica custom come per l'evaluate."""
    print(f" train -> {len(records)} client coinvolti")
    return aggregate_metricrecords(records, weighting_metric_name)  #aggregate_metricrecords è la funzione DI FLOWER


def crea_valutazione_e_selezione_modello(run_config: dict):
    """Costruisce due funzioni che condividono uno stato interno:

    - aggrega_evaluate: passata a FedProx come evaluate_metrics_aggr_fn. Gira ogni round subito
      dopo che i client hanno valutato sul proprio VALIDATION set locale (vedi client_app.py),
      aggrega le statistiche additive e scrive l'mse appena calcolata nello stato condiviso.
    - seleziona_modello_migliore: passata a strategy.start() come evaluate_fn. Flower la chiama
      SEMPRE dopo aggrega_evaluate, nello stesso round, sugli stessi pesi appena aggregati (vedi
      flwr/serverapp/strategy/strategy.py) - quindi puo' limitarsi a leggere l'mse appena scritta
      invece di rivalutare nulla, e decidere se questo round e' il nuovo migliore. Il round 0
      (pesi random, nessun client ha ancora validato) si esclude da solo: a quel punto lo stato
      e' ancora "non fresco".
    """
    stato = {"mse": None, "fresh": False}  # ponte round-per-round tra le due funzioni
    migliore = {"mse": float("inf"), "arrays": None, "round": None}  # miglior checkpoint visto finora

    def aggrega_evaluate(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
        """Aggrega le metriche di evaluate senza usare la media pesata generica di Flower per
        rmse/r2/mae: non sono lineari, quindi mediare i valori già
        calcolati da ogni client non equivale a calcolarli sull'unione dei dati di tutti i client
        (esempio: due client con rmse locale 1 e 3 sullo stesso numero di finestre non danno un
        rmse globale di 2, ma sqrt(5) ≈ 2.236).

        Ogni client manda invece statistiche additive. Qui vengono sommate su tutti i client
        coinvolti in questo round, e le metriche finali vengono calcolate una sola volta, come
        se il modello fosse stato valutato su tutto il set federato in un unico batch.

        Riusata anche per il round di test finale one-off (vedi main()): la scrittura su `stato`
        che fa in quel caso e' innocua, perche' a quel punto nessuno lo legge più."""
        print(f" evaluate -> {len(records)} client coinvolti")

        total_sse = total_sum_abs_error = total_sum_y = total_sum_y_sq = 0.0
        total_num_values = 0
        for record in records:
            metricrecord = next(iter(record.metric_records.values()))
            total_sse += metricrecord["sse"]
            total_sum_abs_error += metricrecord["sum_abs_error"]
            total_sum_y += metricrecord["sum_y"]
            total_sum_y_sq += metricrecord["sum_y_sq"]
            total_num_values += metricrecord["num_values"]

        mse, rmse, r2, mae = metriche_da_statistiche_additive(
            total_sse, total_sum_abs_error, total_sum_y, total_sum_y_sq, total_num_values
        )
        stato["mse"] = mse  # la legge seleziona_modello_migliore, chiamata da Flower subito dopo
        stato["fresh"] = True
        return MetricRecord(
            {"mse": mse, "rmse": rmse, "r2": r2, "mae": mae, "num_values": total_num_values}
        )

    def seleziona_modello_migliore(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        if not stato["fresh"]:  # round 0, o un round senza client validi (fraction-evaluate=0)
            return None
        mse = stato["mse"]
        stato["fresh"] = False  # consumata: il prossimo round deve scrivere di nuovo prima di essere letto

        if mse < migliore["mse"]:
            migliore["mse"] = mse
            migliore["round"] = server_round
            # Copia indipendente dei pesi: aggregate_arrayrecords crea un ArrayRecord nuovo a
            # ogni round, ma ricostruirlo qui evita di dipendere da quella garanzia implicita.
            migliore["arrays"] = ArrayRecord(arrays.to_torch_state_dict())
            print(f" nuovo miglior modello -> round {server_round}: mse validazione={mse:.5f}")
        return None

    return aggrega_evaluate, seleziona_modello_migliore, migliore


@app.main() #grid: Grid è il modo in cui il server vede e raggiunge i nodi disponibili.
def main(grid: Grid, context: Context):
    """Crea il modello globale iniziale, configura la strategia, esegue tutti i round e salva
    il risultato. Chiamata una volta sola: quando ritorna (nulla), l'esperimento è finito."""
    

    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    min_available_clients: int = int(context.run_config["min-available-clients"])

    # Il modello globale viene inizializzato UNA SOLA VOLTA, qui. Il seed va fissato solo in
    # questo punto: garantisce che l'inizializzazione dei pesi sia riproducibile tra run, senza
    # azzerare l'apprendimento a ogni round (cosa che accadrebbe se il seed fosse nel client).

    seed = int(context.run_config["random-state"])
    torch.manual_seed(seed) # serve per rendere riproducibile l'inizializzazione dei pesi del modello globale (e quindi anche dei modelli locali, che partono dai pesi globali)
    global_model = build_model(context.run_config)
    arrays = ArrayRecord(global_model.state_dict())

    # FedProx = FedAvg + un termine prossimale lato client
    # che frena quanto ogni modello locale si allontana dai pesi globali durante il round.
    # Con proximal-mu=0.0 il termine e' sempre nullo: FedAvg e' semplicemente il caso
    # speciale mu=0, non serve mantenere due strategie/due percorsi di codice separati.
    proximal_mu = float(context.run_config["proximal-mu"])

    # aggrega_evaluate valuta ogni round sul VALIDATION set federato (vedi client_app.py);
    # seleziona_modello_migliore legge quella mse (stesso round, stessi pesi - garantito da
    # Flower, vedi docstring sopra) e tiene in memoria il checkpoint a mse piu' bassa vista finora.
    aggrega_evaluate, seleziona_modello_migliore, migliore = crea_valutazione_e_selezione_modello(context.run_config)

    # "num-examples" (numero di finestre locali) è la chiave con cui FedAvg pesa sia
    # l'aggregazione dei pesi del modello sia quella delle metriche.
    strategy = FedProx(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_available_clients,
        min_evaluate_nodes=min_available_clients if fraction_evaluate > 0.0 else 0,
        min_available_nodes=min_available_clients,
        weighted_by_key="num-examples", # serve per pesare l'aggregazione dei pesi e delle metriche in base al numero di finestre locali di ogni istituzione (è di default)
        proximal_mu=proximal_mu,

        # train_mse è additiva (media pesata generica di Flower va bene); rmse/r2/mae
        # dell'evaluate no, quindi aggrega_evaluate le ricalcola da statistiche additive
        train_metrics_aggr_fn=aggrega_train,
        evaluate_metrics_aggr_fn=aggrega_evaluate,
    )

    # esegue l'intero esperimento: tutto il ciclo di training federato, con i round di
    # training e di validazione, viene gestito da start(). Il modello finale utile non e'
    # result.arrays (l'ultimo round), ma migliore["arrays"] (il migliore in validazione).
    strategy.start(
        grid=grid,
        initial_arrays=arrays,  # inizializza il modello globale con i pesi random
        train_config=ConfigRecord({"lr": lr}),  #  la configurazione allegata a ogni messaggio di training
        num_rounds=num_rounds,
        evaluate_fn=seleziona_modello_migliore,  # NON e' una vera valutazione: legge la validazione gia' calcolata e aggiorna il checkpoint migliore
    )

    if migliore["arrays"] is None:  # non e' mai arrivata una validazione valida in nessun round
        raise RuntimeError(
            "Nessun modello selezionato: nessun round ha completato la validazione federata. "
            "Controlla fraction-evaluate (deve essere > 0) e min-available-clients."
        )
    print(f"\nModello migliore: round {migliore['round']} (mse validazione={migliore['mse']:.5f})")

    # Round di TEST finale, one-off, fuori dal ciclo normale: manda il checkpoint migliore (non
    # l'ultimo) a tutti i client con eval-split="test", cosi' il test resta isolato fino a qui.
    # Stesso MessageType.EVALUATE che usa internamente Flower per configure_evaluate: arriva
    # allo stesso handler @app.evaluate() del client, senza bisogno di un tipo di messaggio nuovo.
    record_test = RecordDict({"arrays": migliore["arrays"], "config": ConfigRecord({"eval-split": "test"})})
    messaggi_test = [
        Message(content=record_test, message_type=MessageType.EVALUATE, dst_node_id=node_id)
        for node_id in grid.get_node_ids()
    ]
    risposte_test = grid.send_and_receive(messaggi_test, timeout=3600)
    contenuti_validi = [msg.content for msg in risposte_test if not msg.has_error()]  # stesso filtro che usa Flower internamente

    print(f"\nRisultati di test per client (modello del round {migliore['round']}):")
    for contenuto in contenuti_validi:
        metricrecord = next(iter(contenuto.metric_records.values()))
        # Le 5 statistiche additive di UN SOLO client, passate da sole alla stessa formula
        # usata per l'aggregato: danno esattamente l'mse/rmse/r2/mae di quel client, senza
        # bisogno di una formula diversa per il caso "singolo client" (Servono comunque, non
        # medie: r2/rmse/mae non sono lineari, vedi aggrega_evaluate).
        mse, rmse, r2, mae = metriche_da_statistiche_additive(
            metricrecord["sse"], metricrecord["sum_abs_error"],
            metricrecord["sum_y"], metricrecord["sum_y_sq"], metricrecord["num_values"],
        )
        print(
            f"  client {int(metricrecord['partition-id'])} (istituzione {int(metricrecord['institution-id'])}): "
            f"mse={mse:.5f} rmse={rmse:.4f} r2={r2:.4f} mae={mae:.4f}"
        )

    metriche_test = aggrega_evaluate(contenuti_validi, "num-examples")
    print(f"\nTest finale aggregato (sul modello migliore, round {migliore['round']}): {metriche_test}")

    if context.run_config["save-model"]:
        print("\nSalvataggio del modello migliore su disco...")
        torch.save(migliore["arrays"].to_torch_state_dict(), "final_model.pt")
