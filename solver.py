"""
solver.py
CP-SAT (OR-Tools) per Nurse Rostering con modalità:
- GENERA: crea una soluzione (con baseline "R" o matrice attuale)
- REPAIR: minimizza cambiamenti rispetto alla matrice baseline (matrice corrente), rispettando LOCK e ASSENZE.

Nota: questa è una Beta dimostrativa: focalizzata su vincoli hard + obiettivi principali.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional, Any

from ortools.sat.python import cp_model

from models import Scenario, iso


@dataclass
class SolveResult:
    status: str  # "OPTIMAL"|"FEASIBLE"|"INFEASIBLE"|"MODEL_INVALID"|"UNKNOWN"
    objective_value: Optional[int] = None
    schedule: Optional[Dict[str, Dict[str, str]]] = None
    diagnostics: List[str] = None


def _status_name(status: int) -> str:
    if status == cp_model.OPTIMAL:
        return "OPTIMAL"
    if status == cp_model.FEASIBLE:
        return "FEASIBLE"
    if status == cp_model.INFEASIBLE:
        return "INFEASIBLE"
    if status == cp_model.MODEL_INVALID:
        return "MODEL_INVALID"
    return "UNKNOWN"


def quick_diagnostics(scenario: Scenario) -> List[str]:
    """
    Diagnostica "pre-solver" molto semplice, utile in demo per spiegare infeasible.
    Non sostituisce l'analisi completa, ma dà messaggi leggibili.
    """
    scenario.ensure_matrices()
    msgs: List[str] = []
    days = scenario.dates()
    nurses = scenario.nurses
    req = scenario.coverage

    work_states = ["M", "P", "N"]

    def req_for_day(d: date) -> Tuple[int, int, int]:
        """Requisiti di copertura per un giorno (applica override weekend/festivi)."""
        reqM, reqP, reqN = req.min_M, req.min_P, req.min_N
        if scenario.day_is_weekend_or_holiday(d):
            reqM = req.weekend_min_M if req.weekend_min_M is not None else reqM
            reqP = req.weekend_min_P if req.weekend_min_P is not None else reqP
            reqN = req.weekend_min_N if req.weekend_min_N is not None else reqN
        return int(reqM), int(reqP), int(reqN)

    for idx, d in enumerate(days):
        di = iso(d)
        reqM, reqP, reqN = req_for_day(d)

        # Assenze forzate (input) e lock che fissano lavoro
        forced_abs = 0
        locked_counts = {"M": 0, "P": 0, "N": 0}
        eligible_for_night_today = 0

        for n in nurses:
            a = scenario.absences.get(n.id, {}).get(di, "")
            if a:
                forced_abs += 1

            # eleggibilità notti (molto grossolana: se non_fa_notti o max_notti==0, non conta)
            if (not n.non_fa_notti) and (n.max_notti is None or n.max_notti > 0) and (not a):
                eligible_for_night_today += 1

            # lock: se lockato e stato attuale è lavoro, conta
            if scenario.locks.get(n.id, {}).get(di, False):
                s = scenario.schedule.get(n.id, {}).get(di, "R")
                if s in work_states:
                    locked_counts[s] += 1
                # warning: lock di una assenza senza input assenza -> conflitto sicuro
                if s in scenario.absence_codes and not a:
                    msgs.append(
                        f"{di}: {n.id} lockato su '{s}' ma l'assenza non è impostata in tab 'Assenze' (probabile conflitto)."
                    )

        # Check 1: richiesta totale vs disponibili (senza considerare smonto)
        available_for_work = max(0, len(nurses) - forced_abs)
        if (reqM + reqP + reqN) > available_for_work:
            msgs.append(
                f"{di}: richiesta M+P+N={reqM+reqP+reqN} > disponibili={available_for_work} (assenze forzate={forced_abs})."
            )

        # Check 2: con smonto post-notte, il giorno d ha almeno reqN(d-1) persone in 'S' (minimo teorico)
        if idx >= 1:
            _, _, prev_reqN = req_for_day(days[idx - 1])
            available_with_smonto = max(0, len(nurses) - forced_abs - prev_reqN)
            if (reqM + reqP + reqN) > available_with_smonto:
                msgs.append(
                    f"{di}: considerando smonto (almeno {prev_reqN} in S per le N del giorno prima), "
                    f"servono {reqM+reqP+reqN} lavoratori ma ne restano ~{available_with_smonto}. "
                    f"(Suggerimento: aumentare infermieri o ridurre coperture o ridurre N giornaliere.)"
                )

        # Check 3: notti minime vs eleggibili
        if eligible_for_night_today < reqN:
            msgs.append(
                f"{di}: richieste N={reqN} ma eleggibili per N (non assenti / non_fa_notti / max_notti) = {eligible_for_night_today}."
            )

        # Check 4: lock che impongono già troppe N rispetto ai req o che bloccano risorse
        if locked_counts["N"] > reqN:
            msgs.append(f"{di}: lock su N={locked_counts['N']} > N richieste={reqN} (potrebbe impedire la copertura di M/P).")

    return msgs


def solve_schedule(
    scenario: Scenario,
    mode: str = "repair",
    time_limit_s: int = 10,
    random_seed: int = 42,
) -> SolveResult:
    """
    mode:
      - "generate": usa una baseline debole (di default 'R'), ma se esiste una matrice corrente la usa come baseline
      - "repair": baseline = matrice corrente, minimizza cambiamenti sulle celle NON lockate
    """
    scenario.ensure_matrices()
    diagnostics = quick_diagnostics(scenario)

    days = scenario.dates()
    if not days or not scenario.nurses:
        return SolveResult(status="MODEL_INVALID", diagnostics=["Periodo o lista infermieri vuota."])

    nurses = scenario.nurses
    nurse_ids = [n.id for n in nurses]
    day_ids = [iso(d) for d in days]

    # Stati
    work_states = ["M", "P", "N"]
    rest_states = ["R", "S"]
    absence_states = list(scenario.absence_codes)

    # Stati usabili in modello
    all_states = work_states + rest_states + absence_states

    # Se un codice assenza non è in hours, aggiungilo come 0 ore
    shift_hours = dict(scenario.shift_hours)
    for a in absence_states:
        shift_hours.setdefault(a, 0)

    # Baseline per change-penalty
    baseline = scenario.schedule  # matrice corrente
    # in generate, se non c'è baseline significativa, è già piena di "R"
    # (Scenario.ensure_matrices mette "R" ovunque)

    # Modello
    model = cp_model.CpModel()

    # Variabili x[nurse, day, state] in {0,1}
    x: Dict[Tuple[str, str, str], cp_model.IntVar] = {}
    for nid in nurse_ids:
        for di in day_ids:
            # Se assenza forzata in input: l'unico stato consentito è quel codice.
            forced_abs = scenario.absences.get(nid, {}).get(di, "")
            allowed_states = all_states
            if forced_abs:
                allowed_states = [forced_abs]  # forza
            for s in all_states:
                if s in allowed_states:
                    x[(nid, di, s)] = model.NewBoolVar(f"x_{nid}_{di}_{s}")
                else:
                    # stato non consentito: usa variabile fissa a 0 (modellazione semplice)
                    x[(nid, di, s)] = model.NewConstant(0)

            # Un solo stato al giorno
            model.Add(sum(x[(nid, di, s)] for s in all_states) == 1)

            # Se non è un'assenza forzata, vietiamo che il solver "inventi" assenze (MAL/104/...)
            # -> l'assenza deve arrivare dall'input (o da edit manuale, che aggiornerebbe absences)
            if not forced_abs:
                for a in absence_states:
                    model.Add(x[(nid, di, a)] == 0)

            # Locks: se la cella è lockata, deve restare al valore corrente (baseline)
            if scenario.locks.get(nid, {}).get(di, False):
                locked_state = scenario.schedule.get(nid, {}).get(di, "R") or "R"
                # Se lock confligge con assenza forzata, il modello diventerà infeasible (ok, da segnalare)
                for s in all_states:
                    if s == locked_state:
                        model.Add(x[(nid, di, s)] == 1)
                    else:
                        model.Add(x[(nid, di, s)] == 0)

    # Copertura minima giornaliera
    cov = scenario.coverage
    for d in days:
        di = iso(d)
        reqM, reqP, reqN = cov.min_M, cov.min_P, cov.min_N
        if scenario.day_is_weekend_or_holiday(d):
            reqM = cov.weekend_min_M if cov.weekend_min_M is not None else reqM
            reqP = cov.weekend_min_P if cov.weekend_min_P is not None else reqP
            reqN = cov.weekend_min_N if cov.weekend_min_N is not None else reqN

        model.Add(sum(x[(nid, di, "M")] for nid in nurse_ids) >= reqM)
        model.Add(sum(x[(nid, di, "P")] for nid in nurse_ids) >= reqP)
        model.Add(sum(x[(nid, di, "N")] for nid in nurse_ids) >= reqN)

    # Vincoli specifici infermiere (non fa notti / max notti)
    for n in nurses:
        if n.non_fa_notti:
            for di in day_ids:
                model.Add(x[(n.id, di, "N")] == 0)
        if n.max_notti is not None:
            model.Add(sum(x[(n.id, di, "N")] for di in day_ids) <= n.max_notti)

    # Smonto dopo Notte: se d è N allora d+1 deve essere S.
    # Nota Beta: sull'ultimo giorno *non* imponiamo lo smonto (cadrebbe fuori orizzonte).
    # Questo evita infeasibility quando la copertura richiede N anche nell'ultimo giorno.
    for n in nurses:
        for idx in range(len(days) - 1):
            d = days[idx]
            d1 = days[idx + 1]
            di, di1 = iso(d), iso(d1)
            # N -> S
            model.Add(x[(n.id, di1, "S")] == 1).OnlyEnforceIf(x[(n.id, di, "N")])


    # Transizioni vietate (11h o regole custom): x[a] -> not x[b] il giorno dopo
    forbidden = list(scenario.rules.forbidden_transitions)
    # Nota: alcune transizioni sono implicitamente gestite (N->S) ma qui lasciamo la possibilità di vietare altro.
    for n in nurses:
        for idx in range(len(days) - 1):
            di, di1 = iso(days[idx]), iso(days[idx + 1])
            for a, b in forbidden:
                if a not in all_states or b not in all_states:
                    continue
                # Se oggi è a, domani non può essere b
                model.Add(x[(n.id, di1, b)] == 0).OnlyEnforceIf(x[(n.id, di, a)])

    # Riposo settimanale: in ogni finestra di N giorni (default 7) almeno 1 "rest-like"
    win = scenario.rules.finestra_riposo
    restlike = set(["R"]) | set(absence_states)
    if scenario.rules.s_conta_come_riposo:
        restlike.add("S")

    for n in nurses:
        for start in range(0, len(days) - win + 1):
            dias = [iso(days[start + k]) for k in range(win)]
            model.Add(sum(x[(n.id, di, s)] for di in dias for s in restlike) >= 1)

    # Limite consecutivi lavorati (hard): max_consecutivi_lavorati
    max_consec = scenario.rules.max_consecutivi_lavorati
    if max_consec is not None and max_consec >= 1:
        # in ogni finestra di (max_consec+1) giorni, non posso lavorare tutti i giorni
        wlen = max_consec + 1
        for n in nurses:
            for start in range(0, len(days) - wlen + 1):
                dias = [iso(days[start + k]) for k in range(wlen)]
                model.Add(sum(x[(n.id, di, s)] for di in dias for s in work_states) <= max_consec)

        # ---------- OBIETTIVI (soft) ----------
        weights = scenario.weights
        obj_terms: List[cp_model.LinearExpr] = []

        # In GENERA (da zero) NON applichiamo w_change: serve solo in REPAIR.
        w_change_eff = int(weights.w_change) if mode.lower().strip() == "repair" else 0

        # A) Minimizzare cambiamenti rispetto a baseline (REPAIR)
        if w_change_eff > 0:
            for nid in nurse_ids:
                for di in day_ids:
                    # Se lockata o assenza forzata, non la penalizziamo (è hard).
                    if scenario.locks.get(nid, {}).get(di, False):
                        continue
                    if scenario.absences.get(nid, {}).get(di, ""):
                        continue

                    b = (baseline.get(nid, {}).get(di, "R") or "R")
                    if b not in all_states:
                        b = "R"
                    # change = 1 se != baseline  (one-hot: change = 1 - x[b])
                    change = model.NewBoolVar(f"chg_{nid}_{di}")
                    model.Add(change + x[(nid, di, b)] == 1)
                    obj_terms.append(w_change_eff * change)

        # Helper: work_today bool (1 se turno lavorativo M/P/N)
        work_today: Dict[Tuple[str, str], cp_model.IntVar] = {}
        for nid in nurse_ids:
            for di in day_ids:
                wt = model.NewBoolVar(f"work_{nid}_{di}")
                model.Add(wt == sum(x[(nid, di, s)] for s in work_states))
                work_today[(nid, di)] = wt

        # B) Continuità (anti "MRNSP"):
        #    B1) penalizza i cambi Work<->NonWork tra giorni consecutivi (evita alternanze)
        if int(getattr(weights, "w_work_switch", 0)) > 0:
            w = int(weights.w_work_switch)
            for nid in nurse_ids:
                for idx in range(len(days) - 1):
                    di, di1 = day_ids[idx], day_ids[idx + 1]
                    sw = model.NewBoolVar(f"sw_work_{nid}_{di}")
                    # sw = 1 se work(di) != work(di1)
                    model.Add(work_today[(nid, di)] == work_today[(nid, di1)]).OnlyEnforceIf(sw.Not())
                    model.Add(work_today[(nid, di)] != work_today[(nid, di1)]).OnlyEnforceIf(sw)
                    obj_terms.append(w * sw)

        #    B2) penalizza i cambi turno quando si lavora in giorni consecutivi (es. M->P)
        if int(getattr(weights, "w_shift_switch", 0)) > 0:
            w = int(weights.w_shift_switch)
            for nid in nurse_ids:
                for idx in range(len(days) - 1):
                    di, di1 = day_ids[idx], day_ids[idx + 1]

                    both_work = model.NewBoolVar(f"bothwork_{nid}_{di}")
                    model.Add(both_work <= work_today[(nid, di)])
                    model.Add(both_work <= work_today[(nid, di1)])
                    model.Add(both_work >= work_today[(nid, di)] + work_today[(nid, di1)] - 1)

                    same_list = []
                    for s in work_states:
                        same = model.NewBoolVar(f"same_{nid}_{di}_{s}")
                        model.Add(same <= x[(nid, di, s)])
                        model.Add(same <= x[(nid, di1, s)])
                        model.Add(same >= x[(nid, di, s)] + x[(nid, di1, s)] - 1)
                        same_list.append(same)

                    same_int = model.NewIntVar(0, 1, f"samework_{nid}_{di}")
                    model.Add(same_int == sum(same_list))
                    model.Add(same_int <= both_work)  # sicurezza

                    shift_sw = model.NewIntVar(0, 1, f"sw_shift_{nid}_{di}")
                    # se entrambi lavorano e NON è lo stesso turno -> 1
                    model.Add(shift_sw == both_work - same_int)
                    obj_terms.append(w * shift_sw)

        # C) Bilanciare notti tra infermieri (range max-min)
        if weights.w_balance_nights > 0:
            night_counts = []
            for n in nurses:
                c = model.NewIntVar(0, len(days), f"nights_{n.id}")
                model.Add(c == sum(x[(n.id, di, "N")] for di in day_ids))
                night_counts.append(c)
            maxN = model.NewIntVar(0, len(days), "maxN")
            minN = model.NewIntVar(0, len(days), "minN")
            model.AddMaxEquality(maxN, night_counts)
            model.AddMinEquality(minN, night_counts)
            rng = model.NewIntVar(0, len(days), "rngN")
            model.Add(rng == maxN - minN)
            obj_terms.append(int(weights.w_balance_nights) * rng)

        # D) Bilanciare weekend/festivi (range max-min)
        if weights.w_balance_weekend > 0:
            weekend_days = [iso(d) for d in days if scenario.day_is_weekend_or_holiday(d)]
            weekend_counts = []
            for n in nurses:
                c = model.NewIntVar(0, len(weekend_days), f"wk_{n.id}")
                model.Add(c == sum(work_today[(n.id, di)] for di in weekend_days))
                weekend_counts.append(c)
            maxW = model.NewIntVar(0, len(weekend_days), "maxW")
            minW = model.NewIntVar(0, len(weekend_days), "minW")
            model.AddMaxEquality(maxW, weekend_counts)
            model.AddMinEquality(minW, weekend_counts)
            rng = model.NewIntVar(0, len(weekend_days), "rngW")
            model.Add(rng == maxW - minW)
            obj_terms.append(int(weights.w_balance_weekend) * rng)

        # E) Debito orario: minimizza scostamento oltre tolleranza (±X)
        if weights.w_hours_dev > 0:
            tol = max(0, int(scenario.rules.tolleranza_ore))
            max_per_day = max(int(shift_hours.get("M", 0)), int(shift_hours.get("P", 0)), int(shift_hours.get("N", 0)), 0)
            max_hours = max_per_day * len(days)
            for n in nurses:
                h = model.NewIntVar(0, max_hours, f"hours_{n.id}")
                model.Add(
                    h == sum(
                        shift_hours["M"] * x[(n.id, di, "M")] +
                        shift_hours["P"] * x[(n.id, di, "P")] +
                        shift_hours["N"] * x[(n.id, di, "N")]
                        for di in day_ids
                    )
                )
                target = int(n.target_ore_periodo)
                dev_abs = model.NewIntVar(0, max_hours, f"devabs_{n.id}")
                model.AddAbsEquality(dev_abs, h - target)
                if tol > 0:
                    over = model.NewIntVar(0, max_hours, f"over_{n.id}")
                    model.Add(over >= dev_abs - tol)
                    model.Add(over >= 0)
                    obj_terms.append(int(weights.w_hours_dev) * over)
                else:
                    obj_terms.append(int(weights.w_hours_dev) * dev_abs)

        # F) Penalizza over-coverage (extra persone oltre il minimo richiesto)
        if int(getattr(weights, "w_surplus", 0)) > 0:
            w = int(weights.w_surplus)
            cov = scenario.coverage
            for d in days:
                di = iso(d)
                reqM, reqP, reqN = cov.min_M, cov.min_P, cov.min_N
                if scenario.day_is_weekend_or_holiday(d):
                    reqM = cov.weekend_min_M if cov.weekend_min_M is not None else reqM
                    reqP = cov.weekend_min_P if cov.weekend_min_P is not None else reqP
                    reqN = cov.weekend_min_N if cov.weekend_min_N is not None else reqN

                # (abbiamo già vincolo >= req, quindi surplus = assigned - req è non-negativo)
                assignedM = model.NewIntVar(0, len(nurses), f"assM_{di}")
                assignedP = model.NewIntVar(0, len(nurses), f"assP_{di}")
                assignedN = model.NewIntVar(0, len(nurses), f"assN_{di}")
                model.Add(assignedM == sum(x[(nid, di, "M")] for nid in nurse_ids))
                model.Add(assignedP == sum(x[(nid, di, "P")] for nid in nurse_ids))
                model.Add(assignedN == sum(x[(nid, di, "N")] for nid in nurse_ids))

                surM = model.NewIntVar(0, len(nurses), f"surM_{di}")
                surP = model.NewIntVar(0, len(nurses), f"surP_{di}")
                surN = model.NewIntVar(0, len(nurses), f"surN_{di}")
                model.Add(surM == assignedM - int(reqM))
                model.Add(surP == assignedP - int(reqP))
                model.Add(surN == assignedN - int(reqN))

                obj_terms.append(w * surM)
                obj_terms.append(w * surP)
                obj_terms.append(w * surN)

        if obj_terms:
            model.Minimize(sum(obj_terms))

# Risoluzione
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.num_search_workers = 8  # parallelismo (di solito utile)
    # solver.parameters.log_search_progress = True  # per debug

    status = solver.Solve(model)
    status_name = _status_name(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out_schedule: Dict[str, Dict[str, str]] = {nid: {} for nid in nurse_ids}
        for nid in nurse_ids:
            for di in day_ids:
                # trova stato attivo
                chosen = None
                for s in all_states:
                    v = x[(nid, di, s)]
                    if solver.Value(v) == 1:
                        chosen = s
                        break
                if chosen is None:
                    chosen = "R"
                out_schedule[nid][di] = chosen
        return SolveResult(
            status=status_name,
            objective_value=int(solver.ObjectiveValue()) if obj_terms else None,
            schedule=out_schedule,
            diagnostics=diagnostics,
        )

    return SolveResult(status=status_name, diagnostics=diagnostics)
