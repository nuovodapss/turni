"""
storage.py
Persistenza locale su JSON (save/load scenario).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from models import Scenario


def save_scenario_json(path: str, scenario: Scenario) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = scenario.to_dict()
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_scenario_json(path: str) -> Scenario:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Scenario.from_dict(data)
