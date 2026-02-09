"""
app.py
Streamlit UI - Beta dimostrativa Nurse Rostering con:
- input totalmente in-app (niente Excel in input)
- generazione e REPAIR (minimizza cambiamenti rispetto baseline)
- BLOCCA CELLE (lock)
- export Excel e persistenza JSON locale

Avvio:
  streamlit run app.py
"""
from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from models import Scenario, Nurse, build_demo_scenario, iso
from storage import save_scenario_json, load_scenario_json
from solver import solve_schedule
from reporting import (
    schedule_df,
    absences_df,
    locks_df,
    per_nurse_report,
    coverage_report,
    validate_hard_constraints,
)

st.set_page_config(page_title="Beta Nurse Rostering (Repair + Lock)", layout="wide")


# --------------------------
# Helpers DataFrame <-> Scenario
# --------------------------
WORK_STATES = ["M", "P", "N"]
REST_STATES = ["R", "S"]


def _get_day_ids(sc: Scenario) -> List[str]:
    return [iso(d) for d in sc.dates()]


def df_to_absences(sc: Scenario, df: pd.DataFrame) -> None:
    """Aggiorna sc.absences da DF (index nurse_id, columns day_iso)."""
    day_ids = _get_day_ids(sc)
    for nid in df.index:
        for di in day_ids:
            val = str(df.at[nid, di]) if di in df.columns else ""
            val = "" if val in ("nan", "None", "NA") else val.strip()
            sc.absences[nid][di] = val


def df_to_schedule(sc: Scenario, df: pd.DataFrame) -> None:
    """Aggiorna sc.schedule da DF (index nurse_id, columns day_iso)."""
    day_ids = _get_day_ids(sc)
    allowed_states = WORK_STATES + REST_STATES + list(sc.absence_codes)
    for nid in df.index:
        for di in day_ids:
            val = str(df.at[nid, di]) if di in df.columns else "R"
            val = "R" if val in ("nan", "None", "NA", "") else val.strip()
            if val not in allowed_states:
                val = "R"
            sc.schedule[nid][di] = val


def df_to_locks(sc: Scenario, df: pd.DataFrame) -> None:
    day_ids = _get_day_ids(sc)
    for nid in df.index:
        for di in day_ids:
            sc.locks[nid][di] = bool(df.at[nid, di])


def sync_absences_into_schedule(sc: Scenario) -> None:
    """Se esiste un'assenza (MAL/104/...), forza la schedule in quella cella."""
    day_ids = _get_day_ids(sc)
    for n in sc.nurses:
        for di in day_ids:
            a = sc.absences[n.id][di]
            if a:
                sc.schedule[n.id][di] = a


def sync_schedule_absences_back(sc: Scenario) -> None:
    """
    Se l'utente mette manualmente MAL/104/FERIE/ASP nella matrice turno,
    lo trattiamo come assenza forzata (aggiornando absences).
    """
    day_ids = _get_day_ids(sc)
    abs_set = set(sc.absence_codes)
    for n in sc.nurses:
        for di in day_ids:
            s = sc.schedule[n.id][di]
            if s in abs_set:
                sc.absences[n.id][di] = s
            elif sc.absences[n.id][di] and s not in abs_set:
                # se prima era assenza ma ora è stato messo un turno, annulliamo l'assenza?
                # In beta, scegliamo una regola conservativa:
                # - l'assenza è INPUT HARD -> se esiste in absences, ripristina la schedule a quella assenza.
                sc.schedule[n.id][di] = sc.absences[n.id][di]


def make_excel_bytes(sc: Scenario) -> bytes:
    """Crea un Excel in memoria con 3 fogli: MatriceTurni, Report, Parametri."""
    sc.ensure_matrices()
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_s = schedule_df(sc)
        df_s.to_excel(writer, sheet_name="MatriceTurni")

        rep = per_nurse_report(sc)
        rep.to_excel(writer, sheet_name="Report", index=False)

        cov = coverage_report(sc)
        cov.to_excel(writer, sheet_name="Coperture", index=False)

        # Parametri (semplice key/value)
        params = []
        params.append(("Scenario", sc.nome))
        params.append(("Start", sc.start_date.isoformat()))
        params.append(("End", sc.end_date.isoformat()))
        params.append(("Copertura min M", sc.coverage.min_M))
        params.append(("Copertura min P", sc.coverage.min_P))
        params.append(("Copertura min N", sc.coverage.min_N))
        params.append(("Max consecutivi lavorati", sc.rules.max_consecutivi_lavorati))
        params.append(("Finestra riposo", sc.rules.finestra_riposo))
        params.append(("S conta come riposo", sc.rules.s_conta_come_riposo))
        params.append(("Tolleranza ore", sc.rules.tolleranza_ore))
        params.append(("Forbidden transitions", ", ".join([f"{a}->{b}" for a, b in sc.rules.forbidden_transitions])))
        params.append(("Peso change (repair)", sc.weights.w_change))
        params.append(("Peso continuità Work<->NonWork", sc.weights.w_work_switch))
        params.append(("Peso continuità cambio turno (work->work)", sc.weights.w_shift_switch))
        params.append(("Peso over-coverage (surplus)", sc.weights.w_surplus))
        params.append(("Peso bilanciamento notti", sc.weights.w_balance_nights))
        params.append(("Peso bilanciamento weekend", sc.weights.w_balance_weekend))
        params.append(("Peso ore (dev)", sc.weights.w_hours_dev))
        pd.DataFrame(params, columns=["Parametro", "Valore"]).to_excel(writer, sheet_name="Parametri", index=False)

    return out.getvalue()




def schedule_view_df(sc: Scenario) -> pd.DataFrame:
    """Vista 'umana' della matrice turni (index = Nome, colonna ID)."""
    df = schedule_df(sc).copy()
    id_to_name = {n.id: n.nome for n in sc.nurses}
    df.insert(0, "ID", df.index)
    df.index = [id_to_name.get(i, i) for i in df.index]
    return df

def absences_view_df(sc: Scenario) -> pd.DataFrame:
    df = absences_df(sc).copy()
    id_to_name = {n.id: n.nome for n in sc.nurses}
    df.insert(0, "ID", df.index)
    df.index = [id_to_name.get(i, i) for i in df.index]
    return df

# --------------------------
# State init
# --------------------------
if "scenario" not in st.session_state:
    st.session_state.scenario = build_demo_scenario()
if "last_solve" not in st.session_state:
    st.session_state.last_solve = None

sc: Scenario = st.session_state.scenario
sc.ensure_matrices()

# --------------------------
# Sidebar: scenario save/load
# --------------------------
st.sidebar.header("Scenario (JSON locale)")
scenarios_dir = Path("scenarios")
scenarios_dir.mkdir(exist_ok=True)

existing = sorted([p.name for p in scenarios_dir.glob("*.json")])
sel = st.sidebar.selectbox("Carica da file (cartella ./scenarios)", options=[""] + existing, index=0)

# In Streamlit Cloud (o in demo con repository) può essere comodo caricare un JSON dal pc.
up = st.sidebar.file_uploader("Carica scenario JSON (upload)", type=["json"])
if up is not None:
    try:
        data = json.loads(up.getvalue().decode("utf-8"))
        st.session_state.scenario = Scenario.from_dict(data)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Errore upload JSON: {e}")
colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Carica selezionato", use_container_width=True, disabled=(sel == "")):
        try:
            sc = load_scenario_json(str(scenarios_dir / sel))
            st.session_state.scenario = sc
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Errore nel caricamento: {e}")
with colB:
    if st.button("Carica esempio", use_container_width=True):
        st.session_state.scenario = build_demo_scenario()
        st.rerun()

st.sidebar.divider()
save_name = st.sidebar.text_input("Nome file per salvataggio", value="demo.json")
if st.sidebar.button("Salva scenario", use_container_width=True):
    try:
        save_scenario_json(str(scenarios_dir / save_name), sc)
        st.sidebar.success(f"Salvato in: scenarios/{save_name}")
    except Exception as e:
        st.sidebar.error(f"Errore nel salvataggio: {e}")

# Download (utile in cloud dove non si vede il filesystem)
st.sidebar.download_button(
    "Scarica scenario JSON",
    data=json.dumps(sc.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=save_name if save_name.strip() else "scenario.json",
    mime="application/json",
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.caption("Suggerimento: in demo mostra ‘Carica esempio’, poi fai REPAIR dopo aver messo alcune assenze e lock.")


# --------------------------
# Header + legenda
# --------------------------
st.title("🧑‍⚕️ Beta Nurse Rostering — Genera + Repair + Lock")
st.caption(
    "Obiettivo chiave: **mantenere la matrice**. Ogni ricalcolo usa una modalità **REPAIR** che "
    "minimizza i cambiamenti rispetto alla matrice baseline (quella già costruita / editata)."
)

with st.expander("Legenda", expanded=True):
    st.markdown(
        """
- **M** = Mattina (7h) · **P** = Pomeriggio (7h) · **N** = Notte (10h)  
- **R** = Riposo · **S** = Smonto post-notte  
- Assenze (0h): **MAL**, **104**, **FERIE**, **ASP** (configurabili)  
- **LOCK**: una cella bloccata diventa vincolo HARD nei ricalcoli.
        """
    )

tabs = st.tabs(["1) Setup", "2) Assenze", "3) Generazione", "4) Review & Lock", "5) Report & Validazione"])

# --------------------------
# TAB 1 — Setup
# --------------------------
with tabs[0]:
    st.subheader("Setup periodo, infermieri, coperture, regole, pesi obiettivo")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        sd = st.date_input("Data inizio", value=sc.start_date, key="sd")
    with c2:
        ed = st.date_input("Data fine", value=sc.end_date, key="ed")
    with c3:
        sc.nome = st.text_input("Nome scenario", value=sc.nome)

    if sd != sc.start_date or ed != sc.end_date:
        sc.start_date, sc.end_date = sd, ed
        sc.ensure_matrices()
        st.info("Periodo aggiornato: matrici riallineate (baseline mantenuta per quanto possibile).")

    st.markdown("### Infermieri")
    # NOTE: Streamlit data_editor fa controlli di compatibilità tra dtype pandas e column_config.
    # Se una colonna è configurata come "testo" ma contiene interi (o viceversa) l'app va in errore.
    # Per campi opzionali numerici (es. Max notti) usiamo il dtype "Int64" con NA.
    nurses_df = pd.DataFrame([{
        "id": str(n.id),
        "Nome": str(n.nome),
        "Target ore periodo": int(n.target_ore_periodo),
        "Non fa notti": bool(n.non_fa_notti),
        "Max notti": (n.max_notti if n.max_notti is not None else pd.NA),
    } for n in sc.nurses])

    nurses_df["Max notti"] = nurses_df["Max notti"].astype("Int64")

    nurses_df = st.data_editor(
        nurses_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn(required=True),
            "Nome": st.column_config.TextColumn(required=True),
            "Target ore periodo": st.column_config.NumberColumn(min_value=0, step=1),
            "Non fa notti": st.column_config.CheckboxColumn(),
            "Max notti": st.column_config.NumberColumn(min_value=0, step=1, help="Vuoto = nessun limite"),
        },
        key="nurses_editor",
    )

    # Applica modifiche nurses
    new_nurses = []
    for _, r in nurses_df.iterrows():
        nid = str(r["id"]).strip()
        if not nid:
            continue
        nome = str(r["Nome"]).strip() or nid
        target = int(r["Target ore periodo"] or 0)
        non_notti = bool(r["Non fa notti"])
        maxn = r["Max notti"]
        maxn = None if pd.isna(maxn) else int(maxn)
        new_nurses.append(Nurse(id=nid, nome=nome, target_ore_periodo=target, non_fa_notti=non_notti, max_notti=maxn))
    sc.nurses = new_nurses
    sc.ensure_matrices()

    st.markdown("### Coperture minime (default + weekend/festivi)")
    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
    with cc1:
        sc.coverage.min_M = st.number_input("Min M (default)", min_value=0, value=int(sc.coverage.min_M), step=1)
    with cc2:
        sc.coverage.min_P = st.number_input("Min P (default)", min_value=0, value=int(sc.coverage.min_P), step=1)
    with cc3:
        sc.coverage.min_N = st.number_input("Min N (default)", min_value=0, value=int(sc.coverage.min_N), step=1)
    with cc4:
        sc.coverage.weekend_min_M = st.number_input("Min M (weekend)", min_value=0, value=int(sc.coverage.weekend_min_M or sc.coverage.min_M), step=1)
    with cc5:
        sc.coverage.weekend_min_P = st.number_input("Min P (weekend)", min_value=0, value=int(sc.coverage.weekend_min_P or sc.coverage.min_P), step=1)
    with cc6:
        sc.coverage.weekend_min_N = st.number_input("Min N (weekend)", min_value=0, value=int(sc.coverage.weekend_min_N or sc.coverage.min_N), step=1)

    st.markdown("### Regole (HARD)")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        sc.rules.max_consecutivi_lavorati = st.number_input("Max consecutivi lavorati (M/P/N)", min_value=1, value=int(sc.rules.max_consecutivi_lavorati), step=1)
    with r2:
        sc.rules.s_conta_come_riposo = st.checkbox("S conta come riposo", value=bool(sc.rules.s_conta_come_riposo))
    with r3:
        sc.rules.finestra_riposo = st.number_input("Finestra riposo (giorni)", min_value=2, value=int(sc.rules.finestra_riposo), step=1)
    with r4:
        sc.rules.tolleranza_ore = st.number_input("Tolleranza ore (±X) prima di penalità", min_value=0, value=int(sc.rules.tolleranza_ore), step=1)

    st.markdown("### Transizioni vietate (11h / regole custom)")
    st.caption("Seleziona coppie (oggi→domani) da vietare. Esempio tipico: **P→M**.")
    base_states = ["M", "P", "N", "R", "S"]
    options = [f"{a}→{b}" for a in base_states for b in base_states]
    current = set([f"{a}→{b}" for a, b in sc.rules.forbidden_transitions if a in base_states and b in base_states])
    selected = st.multiselect("Transizioni vietate", options=options, default=sorted(current))
    sc.rules.forbidden_transitions = [(x.split("→")[0], x.split("→")[1]) for x in selected]

    st.markdown("### Festivi extra (oltre Sab/Dom)")
    day_ids = _get_day_ids(sc)
    sc.extra_holidays = st.multiselect("Seleziona date festive extra", options=day_ids, default=sc.extra_holidays)

    st.markdown("### Pesi obiettivo (SOFT) — sliders")
st.caption("Nota: **Genera Turno** ignora il peso A) (w_change). Il peso A) conta soprattutto in **Ricalcola (Repair)** per mantenere la matrice.")
w1, w2, w3, w4, w5 = st.columns(5)
with w1:
    sc.weights.w_change = st.slider("A) Repair: cambia meno possibile", 0, 10000, int(sc.weights.w_change), step=50)
with w2:
    sc.weights.w_work_switch = st.slider("B) Continuità: evita alternanze lavoro/riposo", 0, 200, int(sc.weights.w_work_switch), step=1)
with w3:
    sc.weights.w_shift_switch = st.slider("B2) Continuità: evita cambi turno tra giorni lavorati", 0, 200, int(sc.weights.w_shift_switch), step=1)
with w4:
    sc.weights.w_balance_nights = st.slider("C) Bilancia notti", 0, 200, int(sc.weights.w_balance_nights), step=1)
with w5:
    sc.weights.w_balance_weekend = st.slider("D) Bilancia weekend", 0, 200, int(sc.weights.w_balance_weekend), step=1)

sc.weights.w_hours_dev = st.slider("E) Debito orario (oltre tolleranza)", 0, 500, int(sc.weights.w_hours_dev), step=1)
sc.weights.w_surplus = st.slider("F) Over-coverage: evita extra personale", 0, 50, int(sc.weights.w_surplus), step=1)

# --------------------------
# TAB 2 — Assenze
# --------------------------
with tabs[1]:
    st.subheader("Matrice Disponibilità / Assenze (input in-app)")
    st.caption("Inserisci **solo impedimenti non lavorativi** (MAL/104/FERIE/ASP...). Il solver non può assegnare turni lavorativi su quei giorni.")

    # Bulk apply
    b1, b2, b3, b4, b5 = st.columns([2, 1, 1, 2, 1])
    nurse_opts = ["(Tutti)"] + [f"{n.id} — {n.nome}" for n in sc.nurses]
    with b1:
        seln = st.selectbox("Infermiere per inserimento rapido", options=nurse_opts, index=0)
    with b2:
        dstart = st.selectbox("Dal giorno", options=_get_day_ids(sc), index=0)
    with b3:
        dend = st.selectbox("Al giorno", options=_get_day_ids(sc), index=min(6, len(_get_day_ids(sc)) - 1))
    with b4:
        motivo = st.selectbox("Motivo assenza", options=[""] + list(sc.absence_codes), index=0)
    with b5:
        if st.button("Applica", use_container_width=True, disabled=(motivo == "")):
            ids = [n.id for n in sc.nurses] if seln == "(Tutti)" else [seln.split("—")[0].strip()]
            # applica intervallo
            day_ids = _get_day_ids(sc)
            i0, i1 = day_ids.index(dstart), day_ids.index(dend)
            if i1 < i0:
                i0, i1 = i1, i0
            for nid in ids:
                for di in day_ids[i0:i1+1]:
                    sc.absences[nid][di] = motivo
                    sc.schedule[nid][di] = motivo
                    sc.locks[nid][di] = False  # di default non lockiamo un'assenza; è già hard via absences
            st.success("Assenza applicata (forzata) e matrice turno aggiornata (REPAIR al prossimo solve).")

    # Editor matrice assenze
    abs_df = pd.DataFrame({"NurseID": [n.id for n in sc.nurses], "Nome": [n.nome for n in sc.nurses]}).set_index("NurseID")
    mat = absences_df(sc)
    abs_df = abs_df.join(mat, how="left")
    day_cols = _get_day_ids(sc)
    col_cfg = {"Nome": st.column_config.TextColumn(disabled=True)}
    for di in day_cols:
        col_cfg[di] = st.column_config.SelectboxColumn(options=[""] + list(sc.absence_codes))

    edited = st.data_editor(
        abs_df,
        use_container_width=True,
        column_config=col_cfg,
        key="abs_editor",
    )

    # Scrivi back (solo colonne giorni)
    edited_days = edited[day_cols].copy()
    df_to_absences(sc, edited_days)
    sync_absences_into_schedule(sc)

    st.markdown("#### Preview matrice turno (con assenze forzate)")
    st.dataframe(schedule_view_df(sc), use_container_width=True)

# --------------------------
# TAB 3 — Generazione
# --------------------------
with tabs[2]:
    st.subheader("Generazione Turno (CP-SAT)")
    st.caption("Premi **Genera Turno**: il solver crea una matrice che rispetta i vincoli hard e ottimizza i soft (incluso REPAIR).")

    g1, g2, g3 = st.columns([1, 1, 3])
    with g1:
        tlim = st.slider("Time limit solver (sec)", 2, 60, 10, step=1)
    with g2:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    with g3:
        st.write("")

    if st.button("🚀 Genera Turno", type="primary"):
        sync_absences_into_schedule(sc)
        with st.spinner("Solver in esecuzione..."):
            res = solve_schedule(sc, mode="generate", time_limit_s=int(tlim), random_seed=int(seed))
        st.session_state.last_solve = res

        if res.status in ("OPTIMAL", "FEASIBLE") and res.schedule:
            sc.schedule = res.schedule
            sync_absences_into_schedule(sc)
            st.success(f"Soluzione trovata: {res.status}" + (f" · objective={res.objective_value}" if res.objective_value is not None else ""))
        else:
            st.error(f"Nessuna soluzione: {res.status}")
            if res.diagnostics:
                st.warning("Possibili cause (check rapido):\n- " + "\n- ".join(res.diagnostics[:8]))

    if st.session_state.last_solve is not None:
        res = st.session_state.last_solve
        st.info(f"Ultimo solve: {res.status}" + (f" · objective={res.objective_value}" if res.objective_value is not None else ""))

    st.markdown("### Matrice turno (vista principale)")
    st.dataframe(schedule_view_df(sc), use_container_width=True)

    st.markdown("### Vista per giorno: copertura raggiunta vs richiesta")
    st.dataframe(coverage_report(sc), use_container_width=True, height=320)

# --------------------------
# TAB 4 — Review & Lock
# --------------------------
with tabs[3]:
    st.subheader("Review & Lock (edit manuale + Repair)")
    st.caption("Puoi editare manualmente la matrice. Le celle **LOCK** diventano vincoli HARD al prossimo Repair.")

    # Editor matrice turno
    sched_base = pd.DataFrame({"NurseID": [n.id for n in sc.nurses], "Nome": [n.nome for n in sc.nurses]}).set_index("NurseID")
    mat = schedule_df(sc)
    sched_base = sched_base.join(mat, how="left")
    day_cols = _get_day_ids(sc)

    allowed_states = WORK_STATES + REST_STATES + list(sc.absence_codes)

    sched_cfg = {"Nome": st.column_config.TextColumn(disabled=True)}
    for di in day_cols:
        sched_cfg[di] = st.column_config.SelectboxColumn(options=allowed_states)

    edited_sched = st.data_editor(
        sched_base,
        use_container_width=True,
        column_config=sched_cfg,
        key="schedule_editor",
    )
    df_to_schedule(sc, edited_sched[day_cols])
    # Se l'utente ha messo assenze nella schedule, rendile "input" (absences)
    sync_schedule_absences_back(sc)
    sync_absences_into_schedule(sc)  # assenze input prevalgono

    st.markdown("### Lock celle")
    st.caption("Suggerimento demo: locka alcune assegnazioni, poi imposta nuove assenze e fai Repair per vedere che cambia il minimo necessario.")

    lock_base = pd.DataFrame({"NurseID": [n.id for n in sc.nurses], "Nome": [n.nome for n in sc.nurses]}).set_index("NurseID")
    lock_base = lock_base.join(locks_df(sc), how="left")
    lock_cfg = {"Nome": st.column_config.TextColumn(disabled=True)}
    for di in day_cols:
        lock_cfg[di] = st.column_config.CheckboxColumn()

    edited_locks = st.data_editor(
        lock_base,
        use_container_width=True,
        column_config=lock_cfg,
        key="lock_editor",
    )
    df_to_locks(sc, edited_locks[day_cols])

    q1, q2, q3, q4 = st.columns([2, 2, 2, 2])
    nurse_ids = [n.id for n in sc.nurses]
    with q1:
        quick_nurse = st.selectbox("Quick lock: NurseID", options=nurse_ids, index=0)
    with q2:
        quick_day = st.selectbox("Quick lock: Giorno", options=day_cols, index=0)
    with q3:
        if st.button("Lock ON", use_container_width=True):
            sc.locks[quick_nurse][quick_day] = True
            st.success("Lock attivato.")
            st.rerun()
    with q4:
        if st.button("Lock OFF", use_container_width=True):
            sc.locks[quick_nurse][quick_day] = False
            st.success("Lock disattivato.")
            st.rerun()

    st.divider()
    rr1, rr2, rr3 = st.columns([1, 1, 4])
    with rr1:
        tlim2 = st.slider("Time limit (sec)", 2, 60, 10, step=1, key="tlim_repair")
    with rr2:
        seed2 = st.number_input("Seed", min_value=0, value=42, step=1, key="seed_repair")
    with rr3:
        st.write("")

    if st.button("🔧 Ricalcola (Repair)", type="primary"):
        sync_absences_into_schedule(sc)
        with st.spinner("Repair in esecuzione..."):
            res = solve_schedule(sc, mode="repair", time_limit_s=int(tlim2), random_seed=int(seed2))
        st.session_state.last_solve = res

        if res.status in ("OPTIMAL", "FEASIBLE") and res.schedule:
            sc.schedule = res.schedule
            sync_absences_into_schedule(sc)
            st.success(f"Repair OK: {res.status}" + (f" · objective={res.objective_value}" if res.objective_value is not None else ""))
        else:
            st.error(f"Repair fallito: {res.status}")
            if res.diagnostics:
                st.warning("Possibili cause (check rapido):\n- " + "\n- ".join(res.diagnostics[:8]))

    st.markdown("### Matrice turno aggiornata")
    st.dataframe(schedule_view_df(sc), use_container_width=True)

# --------------------------
# TAB 5 — Report & Validazione
# --------------------------
with tabs[4]:
    st.subheader("Report & Validazione")

    # Diagnostica ore: confronto target vs ore minime richieste dalla copertura
    # (Se i target sono molto più alti del fabbisogno minimo, il solver dovrebbe over-staffare
    #  per raggiungerli, oppure vedrai scostamenti negativi.)
    total_target = sum(int(n.target_ore_periodo) for n in sc.nurses)
    # ore minime richieste dalla copertura (senza contare smonto, che è 0 ore)
    min_hours = 0
    for d in sc.dates():
        reqM, reqP, reqN = sc.coverage.min_M, sc.coverage.min_P, sc.coverage.min_N
        if sc.day_is_weekend_or_holiday(d):
            reqM = sc.coverage.weekend_min_M if sc.coverage.weekend_min_M is not None else reqM
            reqP = sc.coverage.weekend_min_P if sc.coverage.weekend_min_P is not None else reqP
            reqN = sc.coverage.weekend_min_N if sc.coverage.weekend_min_N is not None else reqN
        min_hours += int(reqM) * int(sc.shift_hours.get("M", 0)) + int(reqP) * int(sc.shift_hours.get("P", 0)) + int(reqN) * int(sc.shift_hours.get("N", 0))

    st.info(f"Target ore totali = **{total_target}** · Ore minime richieste dalla copertura = **{min_hours}**. "
            f"Se i target sono molto più alti del fabbisogno, per raggiungerli serve over-coverage (peso F basso) "
            f"oppure coperture più alte.")

    rep = per_nurse_report(sc)
    st.markdown("### Report per infermiere")
    st.dataframe(rep, use_container_width=True, height=360)

    st.markdown("### Coperture per giorno")
    cov = coverage_report(sc)
    st.dataframe(cov, use_container_width=True, height=360)

    st.markdown("### Validazione vincoli (HARD)")
    errs = validate_hard_constraints(sc)
    if not errs:
        st.success("OK: nessuna violazione rilevata dai check.")
    else:
        st.error(f"Trovate {len(errs)} anomalie:")
        st.write("\n".join([f"- {e}" for e in errs[:50]]))
        if len(errs) > 50:
            st.write(f"... (mostrati primi 50)")

    st.divider()
    st.markdown("### Export Excel")
    xls_bytes = make_excel_bytes(sc)
    st.download_button(
        "📤 Esporta Excel (MatriceTurni + Report + Parametri)",
        data=xls_bytes,
        file_name=f"{sc.nome.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("### Salvataggio rapido scenario (JSON)")
    st.caption("Il salvataggio completo include setup, assenze, matrice turno, lock e pesi obiettivo.")
    if st.button("Salva scenario ora", use_container_width=True):
        save_scenario_json(str(Path("scenarios") / "autosave.json"), sc)
        st.success("Salvato in scenarios/autosave.json")
