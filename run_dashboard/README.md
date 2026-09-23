# Run Dashboard

Dashboard Flask read-only per esplorare i run prodotti dalla pipeline EEG-MS.

## Avvio

Dalla root del progetto:

```bash
.venv/bin/python -m pip install -r run_dashboard/requirements.txt
.venv/bin/python run_dashboard/app.py
```

Aprire quindi:

```text
http://127.0.0.1:5000
```

In alternativa, dalla cartella `run_dashboard`:

```bash
cd run_dashboard
../.venv/bin/python app.py
```

La dashboard legge automaticamente:

```text
../artifacts/runs/
```

Per vedere nuovi run, eseguire la pipeline e aggiornare la pagina del browser. L'app non modifica i dati e non rilancia gli esperimenti.
