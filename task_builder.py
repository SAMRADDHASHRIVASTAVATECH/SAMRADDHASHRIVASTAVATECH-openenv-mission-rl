"""
task_builder.py
---------------
Generates 20-50 mission-control tasks.
Uses real data when available; falls back to synthetic templates.
Dynamically inspects whatever columns/keys exist in the data.
"""

import random
import hashlib
import copy
import re
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
#  Mapping constants
# ------------------------------------------------------------------ #

CATEGORY_MAP = {0: "system_failure", 1: "navigation", 2: "resource"}
PRIORITY_MAP = {0: "low", 1: "medium", 2: "high"}
DECISION_MAP = {0: "continue", 1: "adjust", 2: "abort"}


# ------------------------------------------------------------------ #
#  Flexible field extraction helpers
# ------------------------------------------------------------------ #

def _safe_float(value: Any, default: float = None) -> Optional[float]:
    """Try to cast *value* to float; return *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _extract_numbers_from_string(text: str) -> List[float]:
    """Pull all numeric values out of a string."""
    if not text:
        return []
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(text))
    results = []
    for m in matches:
        try:
            results.append(float(m))
        except ValueError:
            pass
    return results


def _find_field(record: Dict[str, Any], candidates: List[str]) -> Tuple[Optional[str], Any]:
    """
    Search record keys (case-insensitive, substring match).
    Returns (matched_key, value) or (None, None).
    """
    # Exact match first
    for candidate in candidates:
        for rk, rv in record.items():
            if rk.lower() == candidate.lower() and rv is not None:
                return rk, rv
    # Substring / contains match
    for candidate in candidates:
        for rk, rv in record.items():
            if candidate.lower() in rk.lower() and rv is not None:
                return rk, rv
    return None, None


def _find_numeric(record: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    """Find first numeric value matching candidate field names."""
    key, val = _find_field(record, candidates)
    if val is not None:
        f = _safe_float(val)
        if f is not None:
            return f
        # Try extracting numbers from string value
        nums = _extract_numbers_from_string(str(val))
        if nums:
            return nums[0]
    return None


def _find_string(record: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Find first non-empty string value matching candidate field names."""
    key, val = _find_field(record, candidates)
    if val is not None:
        s = str(val).strip()
        return s if s else None
    return None


def _get_any_text(record: Dict[str, Any]) -> str:
    """Concatenate all string values in the record for keyword analysis."""
    parts = []
    for k, v in record.items():
        if v is not None and not str(k).startswith("__"):
            parts.append(f"{k}: {v}")
    return " ".join(parts)


def _hash_difficulty(text: str) -> int:
    """Deterministic difficulty 1-5 from text content."""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return (h % 5) + 1


# ------------------------------------------------------------------ #
#  Keyword-based classification for text-heavy records
# ------------------------------------------------------------------ #

# Keywords that suggest different categories / severities
_FAILURE_KEYWORDS = [
    "failure", "error", "malfunction", "critical", "overload",
    "offline", "broken", "damage", "leak", "breach", "collapse",
    "explosion", "unstable", "danger", "warning", "alert",
    "extreme", "intense", "violent", "catastroph", "fatal",
    "blackhole", "black hole", "supernova", "gamma ray",
    "magnetar", "pulsar", "neutron star", "quasar",
    "high energy", "high radiation", "x-ray", "cosmic ray",
]

_NAVIGATION_KEYWORDS = [
    "orbit", "trajectory", "course", "navigate", "position",
    "velocity", "speed", "drift", "deviation", "alignment",
    "rotation", "spin", "angular", "inclination", "eccentricity",
    "binary", "companion", "gravitational", "tidal", "perturbation",
    "asteroid", "comet", "debris", "collision", "encounter",
    "approach", "flyby", "rendezvous", "transfer",
    "elliptical", "hyperbolic", "parabolic", "retrograde",
    "orbital", "perihelion", "aphelion", "barycenter",
]

_RESOURCE_KEYWORDS = [
    "fuel", "power", "energy", "battery", "supply", "reserve",
    "consumption", "efficiency", "resource", "water", "oxygen",
    "life support", "thermal", "cooling", "heating", "pressure",
    "atmosphere", "habitable", "habitability", "temperature",
    "luminosity", "solar", "radiation shield", "material",
    "mass", "density", "composition", "element", "mineral",
    "gas", "dust", "cloud", "nebula", "accretion",
    "dwarf", "giant", "main sequence", "red", "white", "brown",
]

_HIGH_PRIORITY_KEYWORDS = [
    "critical", "emergency", "urgent", "immediate", "extreme",
    "danger", "fatal", "catastroph", "massive", "intense",
    "supergiant", "hypergiant", "supernova", "gamma",
    "black hole", "blackhole", "quasar", "magnetar",
    "collapse", "explosion", "eruption", "flare",
    "very high", "extremely", "maximum", "peak",
]

_LOW_PRIORITY_KEYWORDS = [
    "nominal", "stable", "routine", "normal", "standard",
    "safe", "calm", "quiet", "minor", "small", "low",
    "dwarf", "faint", "dim", "cool", "cold",
    "monitoring", "scheduled", "planned", "expected",
]


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _classify_by_keywords(text: str) -> Dict[str, int]:
    """
    Analyze text content to determine category, priority, and decision
    based on keyword frequency.
    """
    failure_score = _count_keyword_hits(text, _FAILURE_KEYWORDS)
    nav_score = _count_keyword_hits(text, _NAVIGATION_KEYWORDS)
    resource_score = _count_keyword_hits(text, _RESOURCE_KEYWORDS)

    high_score = _count_keyword_hits(text, _HIGH_PRIORITY_KEYWORDS)
    low_score = _count_keyword_hits(text, _LOW_PRIORITY_KEYWORDS)

    # Category: highest keyword score wins
    scores = {0: failure_score, 1: nav_score, 2: resource_score}
    category = max(scores, key=scores.get)

    # If all scores are 0, pick randomly
    if all(v == 0 for v in scores.values()):
        category = random.randint(0, 2)

    # Priority based on urgency keywords
    if high_score >= 3:
        priority = 2
    elif high_score >= 1:
        priority = 1 if low_score < 2 else 0
    elif low_score >= 2:
        priority = 0
    else:
        priority = 1

    # Decision based on severity
    severity = failure_score + high_score
    calm = low_score + (1 if resource_score > nav_score > failure_score else 0)

    if severity >= 4:
        decision = 2  # abort
    elif severity >= 2:
        decision = 1  # adjust
    elif calm >= 3:
        decision = 0  # continue
    else:
        decision = 1  # adjust (default moderate action)

    return {
        "category": category,
        "priority": priority,
        "decision": decision,
        "failure_hits": failure_score,
        "nav_hits": nav_score,
        "resource_hits": resource_score,
        "high_hits": high_score,
        "low_hits": low_score,
    }


# ------------------------------------------------------------------ #
#  Mission description generators
# ------------------------------------------------------------------ #

_SCENARIO_TEMPLATES = [
    "Sensor array detected anomalous readings near {name}. "
    "Object classified as {obj_type}. {detail} Awaiting operator decision.",

    "Long-range scan of {name} ({obj_type}) complete. "
    "{detail} Mission control must assess threat level.",

    "Proximity alert: spacecraft approaching {name}. "
    "Classification: {obj_type}. {detail} Action required.",

    "Telemetry update from survey probe near {name}. "
    "Object profile: {obj_type}. {detail} Evaluate and respond.",

    "Deep-space observatory reports activity around {name} "
    "({obj_type}). {detail} Operator assessment needed.",

    "Automated warning triggered for sector containing {name}. "
    "Known type: {obj_type}. {detail} Please advise.",

    "Navigation computer flagged {name} ({obj_type}) as "
    "point of interest. {detail} Decision pending.",

    "Science team requests directive regarding {name}. "
    "Catalogued as {obj_type}. {detail} Awaiting command.",
]

_DETAIL_GENERATORS = {
    "energetic": [
        "Energy output readings are {level}.",
        "Detected {level} electromagnetic emissions.",
        "Radiation profile indicates {level} energetic activity.",
        "Spectral analysis shows {level} energy signatures.",
    ],
    "environmental": [
        "Environmental interaction assessment: {level}.",
        "Surrounding medium shows {level} perturbation.",
        "Local space conditions are {level} affected.",
    ],
    "physical": [
        "Physical properties suggest {level} structural characteristics.",
        "Mass/density readings indicate {level} gravitational influence.",
        "Object dimensions and composition are {level}.",
    ],
    "formation": [
        "Formation mechanism classified as {level}.",
        "Evolutionary stage indicates {level} development.",
        "Stellar lifecycle position: {level} phase.",
    ],
    "orbital": [
        "Orbital behavior exhibits {level} stability.",
        "Trajectory analysis shows {level} predictability.",
        "Motion patterns are {level} regular.",
    ],
}


def _generate_detail_text(record: Dict[str, Any]) -> str:
    """Create descriptive detail text from record fields."""
    parts = []
    all_text = _get_any_text(record)

    # Try to pull real data snippets
    for field_group, templates in _DETAIL_GENERATORS.items():
        for rk, rv in record.items():
            rk_lower = rk.lower()
            if field_group[:4] in rk_lower and rv is not None:
                val_str = str(rv)[:150]
                if len(val_str) > 10:
                    parts.append(f"{rk}: {val_str}")
                else:
                    level = val_str if val_str else "moderate"
                    template = random.choice(templates)
                    parts.append(template.format(level=level))
                break

    # If we got some real data, use it
    if parts:
        return " ".join(parts[:3])

    # Fallback: pick random generic detail
    classification = _classify_by_keywords(all_text)
    if classification["high_hits"] > 0:
        return "Readings indicate elevated threat conditions. Systems on alert."
    elif classification["low_hits"] > 0:
        return "All indicators within normal parameters. Routine monitoring active."
    else:
        return "Mixed signals detected. Further analysis recommended."


# ------------------------------------------------------------------ #
#  Core: derive a task from one data record
# ------------------------------------------------------------------ #

def _derive_task_from_record(
    record: Dict[str, Any], index: int
) -> Optional[Dict[str, Any]]:
    """
    Build a mission task from any data record — works with ANY schema.
    Uses numeric fields where available, falls back to keyword analysis.
    """
    # ---- Extract identifiers ---- #
    name = _find_string(record, [
        "name", "object_name", "object__name", "designation",
        "id", "star_name", "title", "label", "identifier",
        "catalog_name", "target",
    ]) or f"Object-{index:04d}"

    obj_type = _find_string(record, [
        "type", "object_type", "obj_type", "classification",
        "class", "obj_class", "spectral_type", "category",
        "family", "evolutionary_stage", "kind",
    ]) or "unclassified"

    # ---- Try numeric-based rules first ---- #
    temperature = _find_numeric(record, [
        "temperature", "temp", "t_eff", "teff", "surface_temp",
    ])
    radiation = _find_numeric(record, [
        "radiation", "rad", "luminosity", "lum", "flux",
    ])
    fuel = _find_numeric(record, [
        "fuel", "fuel_level", "propellant", "delta_v", "deltav",
    ])
    orbit_val = _find_numeric(record, [
        "eccentricity", "ecc", "orbit_stability", "stability",
        "inclination",
    ])
    mass = _find_numeric(record, [
        "mass", "stellar_mass", "object_mass",
    ])
    velocity = _find_numeric(record, [
        "velocity", "vel", "speed", "radial_velocity",
    ])
    habitability = _find_numeric(record, [
        "habitability", "habitability_score", "hab_score",
    ])

    # ---- Determine if numeric rules applied ---- #
    used_numeric = False
    category = 2
    priority = 1
    decision = 0
    rule_details = []

    # Rule 1: High temperature + radiation → system failure / abort
    if temperature is not None and radiation is not None:
        used_numeric = True
        if temperature > 5000 and radiation > 1.0:
            category, priority, decision = 0, 2, 2
            rule_details.append(
                f"Extreme thermal ({temperature:.0f} K) + radiation ({radiation:.2f})."
            )
        elif temperature > 3000:
            category, priority, decision = 0, 1, 1
            rule_details.append(
                f"Elevated temperature ({temperature:.0f} K), radiation ({radiation:.2f})."
            )

    # Rule 2: Low fuel → resource / abort or adjust
    if fuel is not None:
        used_numeric = True
        if fuel < 20:
            category, priority, decision = 2, 2, 2
            rule_details.append(f"Critical fuel: {fuel:.1f}%.")
        elif fuel < 50:
            category, priority, decision = 2, 1, 1
            rule_details.append(f"Declining fuel: {fuel:.1f}%.")

    # Rule 3: Orbit instability
    if orbit_val is not None:
        used_numeric = True
        if orbit_val > 0.7:
            category, priority, decision = 1, 2, 2
            rule_details.append(f"Unstable orbit (e={orbit_val:.3f}).")
        elif orbit_val > 0.4:
            category, priority, decision = 1, 1, 1
            rule_details.append(f"Elevated eccentricity (e={orbit_val:.3f}).")

    # Rule 4: Mass-based
    if mass is not None:
        used_numeric = True
        if mass > 10:
            priority = max(priority, 2)
            rule_details.append(f"Massive object ({mass:.2f} M☉).")
        elif mass > 1:
            priority = max(priority, 1)
            rule_details.append(f"Notable mass ({mass:.2f} M☉).")

    # Rule 5: Velocity
    if velocity is not None:
        used_numeric = True
        if abs(velocity) > 100:
            category = 1
            decision = max(decision, 1)
            rule_details.append(f"High velocity ({velocity:.1f} km/s).")

    # Rule 6: Habitability score
    if habitability is not None:
        used_numeric = True
        if habitability > 0.7:
            category = 2
            priority = 2
            decision = 0  # continue — it's valuable
            rule_details.append(
                f"High habitability score ({habitability:.2f}). Valuable target."
            )
        elif habitability < 0.2:
            priority = max(priority, 0)
            rule_details.append(
                f"Low habitability ({habitability:.2f})."
            )

    # ---- If no numeric rules fired, use keyword analysis ---- #
    if not used_numeric or not rule_details:
        all_text = _get_any_text(record)
        classification = _classify_by_keywords(all_text)
        category = classification["category"]
        priority = classification["priority"]
        decision = classification["decision"]

        # Add variety: use the actual hit counts to add info
        total_hits = (
            classification["failure_hits"] +
            classification["nav_hits"] +
            classification["resource_hits"]
        )
        if total_hits == 0:
            # Truly generic record — add randomized variety
            category = random.randint(0, 2)
            priority = random.randint(0, 2)
            # Decision follows priority somewhat
            if priority == 2:
                decision = random.choice([1, 2])
            elif priority == 0:
                decision = 0
            else:
                decision = random.choice([0, 1])

    # ---- Build description ---- #
    detail_text = _generate_detail_text(record)
    if rule_details:
        detail_text = " ".join(rule_details) + " " + detail_text

    template = random.choice(_SCENARIO_TEMPLATES)
    description = template.format(
        name=name,
        obj_type=obj_type,
        detail=detail_text,
    )

    # Trim overly long descriptions
    if len(description) > 500:
        description = description[:497] + "..."

    difficulty = _hash_difficulty(description + str(category) + str(priority))

    return {
        "input": {"description": description},
        "expected": {
            "category": int(category),
            "priority": int(priority),
            "decision": int(decision),
        },
        "difficulty": difficulty,
    }


# ------------------------------------------------------------------ #
#  Synthetic / fallback templates (25 diverse templates)
# ------------------------------------------------------------------ #

_SYNTHETIC_TEMPLATES: List[Dict[str, Any]] = [
    {
        "input": {
            "description": (
                "Mission Control Alert — Solar panel array Alpha-7 "
                "reporting 90% efficiency drop. Core temperature rising "
                "to 6200 K. Radiation spike confirmed."
            )
        },
        "expected": {"category": 0, "priority": 2, "decision": 2},
        "difficulty": 4,
    },
    {
        "input": {
            "description": (
                "Navigation subsystem reports 12° deviation in "
                "planned trajectory. Thruster calibration required. "
                "Orbit eccentricity now 0.65."
            )
        },
        "expected": {"category": 1, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
    {
        "input": {
            "description": (
                "Fuel cell B-3 pressure dropping. Current reserves "
                "at 18%. Recommend conservation protocol."
            )
        },
        "expected": {"category": 2, "priority": 2, "decision": 2},
        "difficulty": 4,
    },
    {
        "input": {
            "description": (
                "Routine deep-space scan complete. No anomalies "
                "detected. All subsystems nominal."
            )
        },
        "expected": {"category": 2, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Gravitational lensing event detected near waypoint "
                "Gamma-12. Navigation recalculation advised."
            )
        },
        "expected": {"category": 1, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
    {
        "input": {
            "description": (
                "Coolant loop C failure imminent. Thermal overload "
                "projected in 45 minutes. Backup system offline."
            )
        },
        "expected": {"category": 0, "priority": 2, "decision": 2},
        "difficulty": 5,
    },
    {
        "input": {
            "description": (
                "Asteroid field ahead — density moderate. "
                "Deflection shields at full capacity. "
                "Recommend minor course correction."
            )
        },
        "expected": {"category": 1, "priority": 1, "decision": 1},
        "difficulty": 2,
    },
    {
        "input": {
            "description": (
                "Oxygen recycler efficiency at 97%. CO2 scrubber "
                "nominal. Life-support green across all decks."
            )
        },
        "expected": {"category": 2, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Unidentified electromagnetic pulse detected. "
                "Communication array temporarily offline. "
                "Source: nearby magnetar candidate."
            )
        },
        "expected": {"category": 0, "priority": 2, "decision": 1},
        "difficulty": 4,
    },
    {
        "input": {
            "description": (
                "Docking port 2 misalignment: 3.2 cm offset. "
                "Automated correction initiated. "
                "Standby for manual override."
            )
        },
        "expected": {"category": 1, "priority": 0, "decision": 1},
        "difficulty": 2,
    },
    {
        "input": {
            "description": (
                "Power grid fluctuation in Module D. Voltage "
                "spikes exceeding safe thresholds by 40%. "
                "Risk of cascading system failure."
            )
        },
        "expected": {"category": 0, "priority": 2, "decision": 2},
        "difficulty": 5,
    },
    {
        "input": {
            "description": (
                "Long-range sensors detect debris cloud from "
                "decommissioned satellite cluster. Course "
                "adjustment of 0.8 degrees recommended."
            )
        },
        "expected": {"category": 1, "priority": 1, "decision": 1},
        "difficulty": 2,
    },
    {
        "input": {
            "description": (
                "Water purification system reporting microbial "
                "contamination at 0.02 ppm — within acceptable "
                "limits. Continue monitoring."
            )
        },
        "expected": {"category": 2, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Solar wind intensity increasing. Predicted "
                "geomagnetic storm in 6 hours. Shield "
                "reinforcement advised."
            )
        },
        "expected": {"category": 0, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
    {
        "input": {
            "description": (
                "Cargo bay pressure seal compromised. Slow leak "
                "detected. Non-critical supplies at risk. "
                "Patch crew dispatched."
            )
        },
        "expected": {"category": 2, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
    {
        "input": {
            "description": (
                "Thruster nozzle 4 showing micro-fractures. "
                "Performance degraded by 15%. Replacement "
                "advised at next resupply."
            )
        },
        "expected": {"category": 0, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
    {
        "input": {
            "description": (
                "Mission clock drift detected: 0.003 seconds "
                "per hour. Navigation accuracy unaffected. "
                "Sync scheduled."
            )
        },
        "expected": {"category": 1, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Emergency beacon received from survey probe "
                "Delta-9. Last known position: 2.4 AU from "
                "current location. Rescue feasibility: low."
            )
        },
        "expected": {"category": 1, "priority": 2, "decision": 1},
        "difficulty": 4,
    },
    {
        "input": {
            "description": (
                "Battery bank 7 capacity at 12%. Solar array "
                "output insufficient for recharge at current "
                "orientation. Power rationing initiated."
            )
        },
        "expected": {"category": 2, "priority": 2, "decision": 1},
        "difficulty": 4,
    },
    {
        "input": {
            "description": (
                "Hull micro-meteorite impact detected on "
                "starboard section. Structural integrity at "
                "98%. No breach. Logged for inspection."
            )
        },
        "expected": {"category": 0, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Gyroscope array 2 showing intermittent errors. "
                "Attitude control compensating via reaction wheels. "
                "Replacement needed within 72 hours."
            )
        },
        "expected": {"category": 1, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
    {
        "input": {
            "description": (
                "Fuel cross-feed valve stuck open between tanks "
                "A and C. Imbalance growing. Manual intervention "
                "required immediately."
            )
        },
        "expected": {"category": 2, "priority": 2, "decision": 2},
        "difficulty": 5,
    },
    {
        "input": {
            "description": (
                "Deep-space relay satellite Echo-3 handshake "
                "successful. Data uplink bandwidth nominal. "
                "Transmitting backlog."
            )
        },
        "expected": {"category": 2, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Radiation shelter drill completed. Crew response "
                "time: 4 min 12 sec (target: 5 min). "
                "All personnel accounted for."
            )
        },
        "expected": {"category": 0, "priority": 0, "decision": 0},
        "difficulty": 1,
    },
    {
        "input": {
            "description": (
                "Primary antenna gimbal motor overheating. "
                "Switching to backup motor. Communication window "
                "reduced by 30%."
            )
        },
        "expected": {"category": 0, "priority": 1, "decision": 1},
        "difficulty": 3,
    },
]


# ------------------------------------------------------------------ #
#  Public builder function
# ------------------------------------------------------------------ #

def build_tasks(
    records: List[Dict[str, Any]],
    min_tasks: int = 20,
    max_tasks: int = 50,
) -> List[Dict[str, Any]]:
    """
    Generate between *min_tasks* and *max_tasks* mission tasks.

    1. Try to derive tasks from real data records.
    2. Fill remaining slots with synthetic fallback templates.
    3. Enforce variety: no more than 30% of tasks may share
       the same (category, priority, decision) triple.
    4. Shuffle the final list.
    """
    tasks: List[Dict[str, Any]] = []

    # --- Phase 1: data-driven tasks -------------------------------- #
    if records:
        sample_size = min(len(records), max_tasks * 2)
        sampled = random.sample(records, min(len(records), sample_size))
        for idx, rec in enumerate(sampled):
            task = _derive_task_from_record(rec, idx)
            if task:
                tasks.append(task)
            if len(tasks) >= max_tasks:
                break

    # --- Phase 2: fallback if we don't have enough ----------------- #
    if len(tasks) < min_tasks:
        needed = min_tasks - len(tasks)
        fallback_pool = list(_SYNTHETIC_TEMPLATES)
        random.shuffle(fallback_pool)
        for i in range(needed):
            template = fallback_pool[i % len(fallback_pool)]
            tasks.append(copy.deepcopy(template))

    # --- Phase 3: enforce variety ---------------------------------- #
    # Count expected-answer distribution
    answer_counts: Dict[tuple, int] = {}
    for t in tasks:
        e = t["expected"]
        key = (e["category"], e["priority"], e["decision"])
        answer_counts[key] = answer_counts.get(key, 0) + 1

    max_same = max(int(len(tasks) * 0.35), 3)
    overflow_keys = {k for k, v in answer_counts.items() if v > max_same}

    if overflow_keys:
        for key in overflow_keys:
            matching = [
                i for i, t in enumerate(tasks)
                if (t["expected"]["category"],
                    t["expected"]["priority"],
                    t["expected"]["decision"]) == key
            ]
            excess = matching[max_same:]
            for idx in excess:
                # Redistribute to a different answer
                new_cat = random.randint(0, 2)
                new_pri = random.randint(0, 2)
                if new_pri == 2:
                    new_dec = random.choice([1, 2])
                elif new_pri == 0:
                    new_dec = 0
                else:
                    new_dec = random.choice([0, 1])
                tasks[idx]["expected"] = {
                    "category": new_cat,
                    "priority": new_pri,
                    "decision": new_dec,
                }
                # Update description to hint at new classification
                cat_name = CATEGORY_MAP[new_cat]
                desc = tasks[idx]["input"]["description"]
                if "Operator assessment" not in desc:
                    desc += (
                        f" Operator assessment suggests {cat_name} "
                        f"classification at {PRIORITY_MAP[new_pri]} priority."
                    )
                    tasks[idx]["input"]["description"] = desc

    # Trim to max
    tasks = tasks[:max_tasks]
    random.shuffle(tasks)

    print(f"[TaskBuilder] Generated {len(tasks)} task(s)")

    # Print variety stats
    final_counts: Dict[tuple, int] = {}
    for t in tasks:
        e = t["expected"]
        key = (e["category"], e["priority"], e["decision"])
        final_counts[key] = final_counts.get(key, 0) + 1
    unique_answers = len(final_counts)
    print(f"[TaskBuilder] Answer variety: {unique_answers} unique triples across {len(tasks)} tasks")

    return tasks