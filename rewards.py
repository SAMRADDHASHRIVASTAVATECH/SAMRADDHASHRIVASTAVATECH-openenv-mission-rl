"""
rewards.py
----------
Translates an evaluation score (0-3) into a numeric reward.
Optionally scales by task difficulty.
"""


# Base reward table (score → reward)
_REWARD_TABLE = {
    3: +10.0,
    2: +5.0,
    1: +2.0,
    0: -3.0,
}


def compute_reward(
    score: int,
    difficulty: int = 1,
    scale_by_difficulty: bool = True,
) -> float:
    """
    Parameters
    ----------
    score               : 0-3  (number of matching action components)
    difficulty           : 1-5  (task difficulty level)
    scale_by_difficulty  : if True, multiply reward by (difficulty / 3)

    Returns
    -------
    float — the computed reward value
    """
    base = _REWARD_TABLE.get(score, -3.0)

    if scale_by_difficulty and difficulty > 0:
        multiplier = difficulty / 3.0
        return round(base * multiplier, 2)

    return base


def reward_summary(score: int, difficulty: int, reward: float) -> str:
    """Human-readable one-liner."""
    return (
        f"Score: {score}/3 | Difficulty: {difficulty} | Reward: {reward:+.2f}"
    )