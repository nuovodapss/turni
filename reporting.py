"""
reporting.py
Calcolo report e validazioni leggibili per la Beta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd

from models import Scenario, iso


WORK_STATES = ["M", "P", "N"]


def schedule_df(scenario: Scenario) -> pd.DataFrame:
    scenario.ensure_matrices()
    days = [iso(d) for d in scenario.dates()]
    idx = [n.id for n in scenario.nurses]
    data = []
    for n in scenario.nurses:
        row = [scenario.schedule[n.id][d] for d in days]
        data.append(row)
    return pd.DataFrame(data, index=idx, columns=days)


def absences_df(scenario: Scenario) -> pd.DataFrame:
    scenario.ensure_matrices()
    days = [iso(d) for d in scenario.dates()]
    idx = [n.id for n in scenario.nurses]
    data = []
    for n in scenario.nurses:
        row = [scenario.absences[n.id][d] for d in days]
        data.append(row)
    return pd.DataFrame(data, index=idx, columns=days)


def locks_df(scenario: Scenario) -> pd.DataFrame:
    scenario.ensure_matrices()
    days = [iso(d) for d in scenario.dates()]
    idx = [n.id for n in scenario.nurses]
    data = []
    for n in scenario.nurses:
        row = [bool(scenario.locks[n.id][d]) for d in days]
        data.append(row)
    return pd.DataFrame(data, index=idx, columns=days)


def _max_consecutive_work(seq: List[str]) -> int:
    mx = 0
    cur = 0
    for s in seq:
        if s in WORK_STATES:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return mx


def _count_work_switches(seq: List[str]) -> int:
    """Conta quante volte si passa Lavoro↔Non lavoro tra giorni consecutivi."""
    def is_work(s: str) -> bool:
        return s in WORK_STATES
    c = 0
    for i in range(len(seq) - 1):
        if is_work(seq[i]) != is_work(seq[i + 1]):
            c += 1
    return c


def _count_shift_switches(seq: List[str]) -> int:
    """Conta i cambi turno tra giorni consecutivi quando entrambi sono lavorativi (M/P/N)."""
    c = 0
    for i in range(len(seq) - 1):
        if seq[i] in WORK_STATES and seq[i + 1] in WORK_STATES and seq[i] != seq[i + 1]:
            c += 1
    return c


def _count_weekend_work(scenario: Scenario, nurse_id: str) -> int:
    c = 0
    for d in scenario.dates():
        di = iso(d)
        if scenario.day_is_weekend_or_holiday(d) and scenario.schedule[nurse_id][di] in WORK_STATES:
            c += 1
    return c


def _weekly_rest_violations(scenario: Scenario, nurse_id: str) -> List[str]:
    """Ritorna lista di finestre (start_iso) che violano il riposo settimanale."""
    days = scenario.dates()
    win = scenario.rules.finestra_riposo
    restlike = {"R"} | set(scenario.absence_codes)
    if scenario.rules.s_conta_come_riposo:
        restlike.add("S")

    viol = []
    for start in range(0, len(days) - win + 1):
        sub = days[start:start + win]
        ok = any(scenario.schedule[nurse_id][iso(d)] in restlike for d in sub)
        if not ok:
            viol.append(f"{iso(sub[0])}→{iso(sub[-1])}")
    return viol


def per_nurse_report(scenario: Scenario) -> pd.DataFrame:
    scenario.ensure_matrices()
    days = [iso(d) for d in scenario.dates()]
    shift_hours = dict(scenario.shift_hours)
    for a in scenario.absence_codes:
        shift_hours.setdefault(a, 0)

    rows = []
    for n in scenario.nurses:
        seq = [scenario.schedule[n.id][d] for d in days]
        ore = sum(shift_hours.get(s, 0) for s in seq)
        notti = sum(1 for s in seq if s == "N")
        wk = _count_weekend_work(scenario, n.id)
        mx_consec = _max_consecutive_work(seq)
        viol_rest = _weekly_rest_violations(scenario, n.id)

        rows.append({
            "Infermiere": n.nome,
            "Ore assegnate": ore,
            "Target ore": n.target_ore_periodo,
            "Scostamento": ore - n.target_ore_periodo,
            "#Notti": notti,
            "#Weekend lavorati": wk,
            "Max consecutivi (M/P/N)": mx_consec,
            "Switch lavoro↔riposo": _count_work_switches(seq),
            "Switch turno (work->work)": _count_shift_switches(seq),
            "Violazioni riposo settimanale": len(viol_rest),
        })

    return pd.DataFrame(rows)


def coverage_report(scenario: Scenario) -> pd.DataFrame:
    scenario.ensure_matrices()
    cov = scenario.coverage
    rows = []
    for d in scenario.dates():
        di = iso(d)
        reqM, reqP, reqN = cov.min_M, cov.min_P, cov.min_N
        if scenario.day_is_weekend_or_holiday(d):
            reqM = cov.weekend_min_M if cov.weekend_min_M is not None else reqM
            reqP = cov.weekend_min_P if cov.weekend_min_P is not None else reqP
            reqN = cov.weekend_min_N if cov.weekend_min_N is not None else reqN

        m = sum(1 for n in scenario.nurses if scenario.schedule[n.id][di] == "M")
        p = sum(1 for n in scenario.nurses if scenario.schedule[n.id][di] == "P")
        nN = sum(1 for n in scenario.nurses if scenario.schedule[n.id][di] == "N")
        rows.append({
            "Giorno": di,
            "M assegnati": m, "M richiesti": reqM, "M OK": m >= reqM,
            "P assegnati": p, "P richiesti": reqP, "P OK": p >= reqP,
            "N assegnati": nN, "N richiesti": reqN, "N OK": nN >= reqN,
            "Totale lav.": m + p + nN,
        })
    return pd.DataFrame(rows)


def validate_hard_constraints(scenario: Scenario) -> List[str]:
    """Validazione semplice dei vincoli hard a posteriori (utile come check demo)."""
    scenario.ensure_matrices()
    errors: List[str] = []
    days = scenario.dates()
    day_ids = [iso(d) for d in days]

    all_states = ["M", "P", "N", "R", "S"] + list(scenario.absence_codes)

    # 1) Un solo stato al giorno (qui è un valore singolo per cella, quindi controlliamo che sia valido)
    for n in scenario.nurses:
        for di in day_ids:
            s = scenario.schedule[n.id][di]
            if s not in all_states:
                errors.append(f"{n.nome} {di}: stato non valido '{s}'.")

    # 2) Copertura minima
    cov_df = coverage_report(scenario)
    for _, r in cov_df.iterrows():
        if not bool(r["M OK"]):
            errors.append(f"{r['Giorno']}: copertura M insufficiente ({r['M assegnati']}/{r['M richiesti']}).")
        if not bool(r["P OK"]):
            errors.append(f"{r['Giorno']}: copertura P insufficiente ({r['P assegnati']}/{r['P richiesti']}).")
        if not bool(r["N OK"]):
            errors.append(f"{r['Giorno']}: copertura N insufficiente ({r['N assegnati']}/{r['N richiesti']}).")

    # 4) Smonto dopo Notte
    for n in scenario.nurses:
        for i in range(len(days) - 1):
            di = day_ids[i]
            di1 = day_ids[i + 1]
            if scenario.schedule[n.id][di] == "N" and scenario.schedule[n.id][di1] != "S":
                errors.append(f"{n.nome} {di}->{di1}: dopo N deve esserci S (trovato {scenario.schedule[n.id][di1]}).")

    # 6) Assenze (se marcato assente, niente M/P/N)
    for n in scenario.nurses:
        for di in day_ids:
            a = scenario.absences[n.id][di]
            if a and scenario.schedule[n.id][di] in WORK_STATES:
                errors.append(f"{n.nome} {di}: assenza {a} ma assegnato {scenario.schedule[n.id][di]}.")

    # 7) Max consecutivi
    maxc = scenario.rules.max_consecutivi_lavorati
    for n in scenario.nurses:
        seq = [scenario.schedule[n.id][di] for di in day_ids]
        if _max_consecutive_work(seq) > maxc:
            errors.append(f"{n.nome}: supera max consecutivi lavorati ({_max_consecutive_work(seq)} > {maxc}).")

    # 5) Riposo settimanale
    for n in scenario.nurses:
        viol = _weekly_rest_violations(scenario, n.id)
        if viol:
            errors.append(f"{n.nome}: violazioni riposo settimanale in finestre: {', '.join(viol[:5])}" + (" ..." if len(viol) > 5 else ""))

    return errors
