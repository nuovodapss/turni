"""
models.py
Modelli dati (Scenario / Infermieri / Parametri) per la Beta di Nurse Rostering.

Obiettivo: mantenere tutto "in-app" (Streamlit) e serializzare su JSON locale.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any


# Turni di default (ore)
DEFAULT_SHIFT_HOURS: Dict[str, int] = {
    "M": 7,
    "P": 7,
    "N": 10,
    "R": 0,   # Riposo
    "S": 0,   # Smonto post-notte
}

# Codici assenza non lavorativi (0 ore)
DEFAULT_ABSENCE_CODES: List[str] = ["MAL", "104", "FERIE", "ASP"]


WORK_STATES_DEFAULT = ["M", "P", "N"]
REST_STATES_DEFAULT = ["R", "S"]


def daterange(start: date, end: date) -> List[date]:
    """Inclusivo su start, inclusivo su end."""
    if end < start:
        return []
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def iso(d: date) -> str:
    return d.isoformat()


@dataclass
class Nurse:
    id: str
    nome: str
    target_ore_periodo: int
    non_fa_notti: bool = False
    max_notti: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Nurse":
        return Nurse(**d)


@dataclass
class CoverageRules:
    """Requisiti di copertura minimi: default + override weekend/festivi."""
    min_M: int = 2
    min_P: int = 2
    min_N: int = 1

    weekend_min_M: Optional[int] = None
    weekend_min_P: Optional[int] = None
    weekend_min_N: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CoverageRules":
        return CoverageRules(**d)


@dataclass
class RuleParams:
    max_consecutivi_lavorati: int = 6
    s_conta_come_riposo: bool = True
    finestra_riposo: int = 7

    # Transizioni vietate tra giorni consecutivi (es. ("P","M"))
    forbidden_transitions: List[Tuple[str, str]] = field(default_factory=list)

    # Tolleranza ore (±X) prima di penalità forte
    tolleranza_ore: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuple -> list per JSON
        d["forbidden_transitions"] = [list(x) for x in self.forbidden_transitions]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RuleParams":
        ft = d.get("forbidden_transitions", [])
        d = dict(d)
        d["forbidden_transitions"] = [tuple(x) for x in ft]
        return RuleParams(**d)


@dataclass
class ObjectiveWeights:
    """
    Pesi obiettivo (SOFT).

    Nota Beta:
    - in modalità **GENERA** (da zero) il solver NON usa di default `w_change`
      (altrimenti, con baseline tutta "R", produrrebbe soluzioni "minime" e poco realistiche).
    - in modalità **REPAIR** `w_change` è il peso principale per "mantenere la matrice".
    """

    # A) mantenere matrice (solo REPAIR)
    w_change: int = 3000

    # Continuità "sensata"
    # - penalizza i cambi Work<->NonWork (evita alternanze tipo M-R-M-R)
    w_work_switch: int = 20
    # - penalizza i cambi turno quando si lavora in giorni consecutivi (es. M->P)
    w_shift_switch: int = 10

    # Equità
    w_balance_nights: int = 30
    w_balance_weekend: int = 20

    # Ore target (deviazione oltre tolleranza)
    w_hours_dev: int = 80

    # Penalità per over-coverage (assegnare più persone del minimo richiesto)
    w_surplus: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectiveWeights":
        """
        Compatibilità con JSON vecchi (v3):
        - ignora chiavi sconosciute
        - tollera campi non presenti
        """
        w = ObjectiveWeights()
        if not d:
            return w

        allowed = set(w.to_dict().keys())
        for k, v in d.items():
            if k in allowed:
                try:
                    setattr(w, k, int(v))
                except Exception:
                    pass

        # mapping best-effort da vecchie chiavi
        # (in v3 c'erano w_adj_work / w_triple_work che spingevano a spezzare la continuità)
        if "w_adj_work" in d and "w_work_switch" not in d:
            # teniamo il default (continuità), non copiamo il vecchio significato
            pass
        if "w_triple_work" in d and "w_shift_switch" not in d:
            pass

        return w


@dataclass
class Scenario:
    nome: str = "scenario"
    start_date: date = field(default_factory=lambda: date.today().replace(day=1))
    end_date: date = field(default_factory=lambda: date.today().replace(day=28))

    nurses: List[Nurse] = field(default_factory=list)
    coverage: CoverageRules = field(default_factory=CoverageRules)
    rules: RuleParams = field(default_factory=RuleParams)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    # Parametri turni e assenze
    shift_hours: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SHIFT_HOURS))
    absence_codes: List[str] = field(default_factory=lambda: list(DEFAULT_ABSENCE_CODES))

    # Giorni festivi extra oltre weekend (lista ISO date)
    extra_holidays: List[str] = field(default_factory=list)

    # Matrice assenze input in-app: absences[nurse_id][day_iso] = code oppure ""
    absences: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Matrice turno corrente: schedule[nurse_id][day_iso] = state (M/P/N/R/S o assenza)
    schedule: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Lock celle: locks[nurse_id][day_iso] = True/False
    locks: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def dates(self) -> List[date]:
        return daterange(self.start_date, self.end_date)

    def day_is_weekend_or_holiday(self, d: date) -> bool:
        if d.weekday() >= 5:
            return True
        return iso(d) in set(self.extra_holidays)

    def ensure_matrices(self) -> None:
        """Inizializza matrici (assenze/schedule/locks) per tutti i nurse e giorni."""
        days = [iso(d) for d in self.dates()]
        for n in self.nurses:
            self.absences.setdefault(n.id, {})
            self.schedule.setdefault(n.id, {})
            self.locks.setdefault(n.id, {})
            for di in days:
                self.absences[n.id].setdefault(di, "")
                self.schedule[n.id].setdefault(di, "R")
                self.locks[n.id].setdefault(di, False)

    def states_all(self) -> List[str]:
        """Lista completa degli stati nel modello."""
        base = list(self.shift_hours.keys())
        # garantisci che i codici assenza siano presenti come stati
        for a in self.absence_codes:
            if a not in base:
                base.append(a)
        return base

    def to_dict(self) -> Dict[str, Any]:
        self.ensure_matrices()
        return {
            "nome": self.nome,
            "start_date": iso(self.start_date),
            "end_date": iso(self.end_date),
            "nurses": [n.to_dict() for n in self.nurses],
            "coverage": self.coverage.to_dict(),
            "rules": self.rules.to_dict(),
            "weights": self.weights.to_dict(),
            "shift_hours": dict(self.shift_hours),
            "absence_codes": list(self.absence_codes),
            "extra_holidays": list(self.extra_holidays),
            "absences": self.absences,
            "schedule": self.schedule,
            "locks": self.locks,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Scenario":
        sc = Scenario(
            nome=d.get("nome", "scenario"),
            start_date=date.fromisoformat(d["start_date"]),
            end_date=date.fromisoformat(d["end_date"]),
            nurses=[Nurse.from_dict(x) for x in d.get("nurses", [])],
            coverage=CoverageRules.from_dict(d.get("coverage", {})),
            rules=RuleParams.from_dict(d.get("rules", {})),
            weights=ObjectiveWeights.from_dict(d.get("weights", {})),
            shift_hours=d.get("shift_hours", dict(DEFAULT_SHIFT_HOURS)),
            absence_codes=d.get("absence_codes", list(DEFAULT_ABSENCE_CODES)),
            extra_holidays=d.get("extra_holidays", []),
            absences=d.get("absences", {}),
            schedule=d.get("schedule", {}),
            locks=d.get("locks", {}),
        )
        sc.ensure_matrices()
        return sc


def build_demo_scenario() -> Scenario:
    """Scenario demo pre-caricato (10-20 infermieri, coperture modeste, alcune assenze)."""
    start = date.today().replace(day=1)
    end = start + timedelta(days=27)  # ~28 giorni
    nurses = []
    for i in range(12):
        nurses.append(
            Nurse(
                id=f"N{i+1:02d}",
                nome=f"Infermiere {i+1:02d}",
                target_ore_periodo=150,  # demo più realistica con coperture più alte
                non_fa_notti=(i in (10, 11)),
                max_notti=6 if i < 10 else 0,
            )
        )

    sc = Scenario(
        nome="Esempio Demo",
        start_date=start,
        end_date=end,
        nurses=nurses,
        coverage=CoverageRules(min_M=3, min_P=3, min_N=2, weekend_min_M=3, weekend_min_P=2, weekend_min_N=2),
        rules=RuleParams(
            max_consecutivi_lavorati=6,
            s_conta_come_riposo=True,
            finestra_riposo=7,
            forbidden_transitions=[
                ("P", "M"),
                ("N", "M"),
                ("N", "P"),
                ("N", "N"),
                ("S", "N")  # esempio: evita smonto->notte immediata (demo)
            ],
            tolleranza_ore=8,
        ),
        weights=ObjectiveWeights(w_change=3000, w_work_switch=20, w_shift_switch=10, w_balance_nights=30, w_balance_weekend=20, w_hours_dev=80, w_surplus=2),
    )
    sc.ensure_matrices()

    # Aggiungi alcune assenze demo
    days = sc.dates()
    if len(days) >= 10:
        sc.absences["N03"][iso(days[3])] = "FERIE"
        sc.absences["N03"][iso(days[4])] = "FERIE"
        sc.absences["N07"][iso(days[10])] = "MAL"
        sc.absences["N11"][iso(days[7])] = "104"
        sc.absences["N12"][iso(days[15])] = "ASP"

    # Baseline iniziale: tutto Riposo
    for n in sc.nurses:
        for d in days:
            sc.schedule[n.id][iso(d)] = "R"
            sc.locks[n.id][iso(d)] = False

    return sc
