"""
evaluator.py
------------
Compares an agent's action tuple against the expected answer.
Returns a score (0-3) indicating how many components matched.
"""

from typing import Dict, Tuple


def evaluate_action(
    action: Tuple[int, int, int],
    expected: Dict[str, int],
) -> Dict[str, object]:
    """
    Parameters
    ----------
    action   : (category, priority, decision)  — the agent's choice
    expected : {"category": int, "priority": int, "decision": int}

    Returns
    -------
    dict with keys:
        score           – int 0-3
        category_match  – bool
        priority_match  – bool
        decision_match  – bool
        details         – str
    """
    cat_match = int(action[0]) == int(expected["category"])
    pri_match = int(action[1]) == int(expected["priority"])
    dec_match = int(action[2]) == int(expected["decision"])

    score = sum([cat_match, pri_match, dec_match])

    details_parts = []
    if cat_match:
        details_parts.append("category ✓")
    else:
        details_parts.append(
            f"category ✗ (got {action[0]}, expected {expected['category']})"
        )
    if pri_match:
        details_parts.append("priority ✓")
    else:
        details_parts.append(
            f"priority ✗ (got {action[1]}, expected {expected['priority']})"
        )
    if dec_match:
        details_parts.append("decision ✓")
    else:
        details_parts.append(
            f"decision ✗ (got {action[2]}, expected {expected['decision']})"
        )

    return {
        "score": score,
        "category_match": cat_match,
        "priority_match": pri_match,
        "decision_match": dec_match,
        "details": " | ".join(details_parts),
    }