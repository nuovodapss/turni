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

### Continuità (anti soluzioni “MRNSP”)
Per evitare matrici “a scacchiera” (alternanze lavoro/riposo o cambi turno inutili), il solver usa due pesi:
- `w_work_switch`: penalizza i cambi **Lavoro ↔ Non lavoro** tra giorni consecutivi (favorisce blocchi)
- `w_shift_switch`: penalizza i cambi **M↔P↔N** quando si lavora in giorni consecutivi (favorisce continuità di turno)

In modalità **Genera Turno**, `w_change` viene ignorato (serve solo nel **Repair**).

### Lock celle
Se una cella è **LOCK**, diventa un vincolo hard: nel prossimo solve quella assegnazione deve rimanere identica.

### Assenze
Le assenze inserite nella tab **Assenze** sono vincoli hard:
- se in un giorno c’è `MAL`/`104`/`FERIE`/`ASP`, quel giorno **non può essere** M/P/N
- la cella viene forzata al codice di assenza

## Nota su INFEASIBLE (molto comune in demo)
Se richiedi **N notti ogni giorno** e hai attivo lo **smonto post-notte (S)**,
ricorda che il giorno successivo almeno una persona sarà in `S` e non potrà coprire M/P.

Regola pratica: con coperture costanti, serve almeno:

`#infermieri >= (min_M + min_P + min_N) + min_N`  → cioè `min_M + min_P + 2*min_N`

Esempio: se chiedi M=2, P=2, N=1, il minimo realistico è **6 infermieri** (non 5).

Inoltre, in questa Beta lo smonto viene imposto solo **se esiste il giorno d+1**:
una `N` nell'ultimo giorno del periodo è ammessa (lo smonto cadrebbe nel periodo successivo).

## Dataset demo
Nella sidebar trovi **Carica esempio** (12 infermieri, coperture modeste, alcune assenze).
È perfetto per fare una demo al capo:
1) genera turni
2) locka qualche cella
3) inserisci nuove assenze
4) premi Repair → vedrai che cambia solo il necessario.

