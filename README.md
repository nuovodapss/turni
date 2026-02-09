# Beta Nurse Rostering — Streamlit + OR-Tools (Repair + Lock)

Questa è una **BETA dimostrativa** (UI in Italiano) per:
- generare una matrice turni infermieri
- **riparare (repair)** una matrice esistente minimizzando i cambiamenti
- gestire **assenze** in-app (MAL/104/FERIE/ASP…)
- bloccare celle (**LOCK**) come vincoli hard
- salvare/caricare scenari in **JSON** locale
- esportare Excel (openpyxl)

## Requisiti
- Python 3.11+
- pacchetti: `streamlit`, `ortools`, `pandas`, `openpyxl`

## Avvio rapido

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Concetti chiave (Repair + Lock)

### Baseline (matrice “attuale”)
La **matrice turno corrente** è la baseline: quando premi **Ricalcola (Repair)**,
il solver cerca una nuova soluzione che rispetti i vincoli hard **cambiando il meno possibile**
rispetto alla baseline (penalità `w_change`).

### Lock celle
Se una cella è **LOCK**, diventa un vincolo hard: nel prossimo solve quella assegnazione deve rimanere identica.

### Assenze
Le assenze inserite nella tab **Assenze** sono vincoli hard:
- se in un giorno c’è `MAL`/`104`/`FERIE`/`ASP`, quel giorno **non può essere** M/P/N
- la cella viene forzata al codice di assenza

## Dataset demo
Nella sidebar trovi **Carica esempio** (12 infermieri, coperture modeste, alcune assenze).
È perfetto per fare una demo al capo:
1) genera turni
2) locka qualche cella
3) inserisci nuove assenze
4) premi Repair → vedrai che cambia solo il necessario.

