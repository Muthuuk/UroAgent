"""Synthetic clinical vignettes for OAB, BPS and UTI (research demonstrator only).

Each case has a free-text note written from templates plus the true label. Symptom
patterns follow broad textbook descriptions and deliberately overlap, so the task is
realistic rather than trivial. No real patient data are used.
"""
import numpy as np
import pandas as pd

CLASSES = ["OAB", "BPS", "UTI"]


def _note(rng, label):
    age = int(rng.integers(19, 85))
    female = rng.random() < (0.85 if label != "OAB" else 0.7)
    parts = [f"{age}-year-old woman" if female else f"{age}-year-old man"]
    freq, noct = int(rng.normal(12.5, 3.5)), int(max(0, rng.normal(2.3, 1.3)))
    say_counts = rng.random() < 0.8
    if label == "OAB":
        dur = rng.choice([f"{int(rng.integers(2, 12))} weeks", f"{int(rng.integers(3, 36))} months"], p=[0.25, 0.75])
        parts.append(rng.choice([f"with urgency for {dur}", f"frequent urgent need to pass urine for {dur}", f"bladder symptoms for {dur}"]))
        if rng.random() < 0.55: parts.append("leaks on the way to the toilet")
        if rng.random() < 0.25: parts.append(f"lower tummy pressure, pain {int(rng.integers(1, 5))}/10")
        if rng.random() < 0.10: parts.append("some burning")
        if rng.random() < 0.25: parts.append("coffee makes it worse")
        if rng.random() < 0.10: parts.append("leukocytes positive on dipstick")
    elif label == "BPS":
        dur = rng.choice([f"{int(rng.integers(6, 12))} weeks", f"{int(rng.integers(2, 60))} months"], p=[0.2, 0.8])
        if rng.random() < 0.8:
            parts.append(f"with bladder pain for {dur}, pain {int(np.clip(rng.normal(5.5, 2), 1, 10))}/10")
        else:
            parts.append(f"with pelvic pressure and frequency for {dur}")
        if rng.random() < 0.55: parts.append(rng.choice(["pain builds up as the bladder fills", "pain eased by passing urine"]))
        if rng.random() < 0.45: parts.append(rng.choice(["flares with coffee and citrus", "spicy food makes it worse"]))
        if rng.random() < 0.45: parts.append("previous urine cultures were negative")
        if rng.random() < 0.5: parts.append("urgency")
        if rng.random() < 0.3: parts.append("burning at times")
        if rng.random() < 0.15: parts.append("leaks occasionally")
        if rng.random() < 0.12: parts.append("leukocytes positive on dipstick")
        if rng.random() < 0.15: parts.append("recurrent UTIs in the past")
    else:  # UTI
        if rng.random() < 0.75:
            parts.append(f"symptoms came on suddenly {int(rng.integers(1, 7))} days ago")
        else:
            parts.append(f"urinary symptoms for {int(rng.integers(2, 6))} weeks")
        if rng.random() < 0.7: parts.append("burning when passing urine")
        if rng.random() < 0.5: parts.append("cloudy smelly urine")
        if rng.random() < 0.55: parts.append("urgency")
        if rng.random() < 0.4: parts.append(f"lower tummy pain {int(rng.integers(2, 7))}/10")
        if rng.random() < 0.45: parts.append("nitrites positive on dipstick")
        if rng.random() < 0.55: parts.append("leukocytes positive")
        if rng.random() < 0.2: parts.append("recurrent UTIs in the past")
        if rng.random() < 0.15: parts.append("leaks occasionally")
        if rng.random() < 0.1: parts.append("coffee makes it worse")
    if say_counts:
        parts.insert(2, f"passing urine {max(freq, 5)} times a day and up {noct} times at night")
    return ", ".join(parts) + "."


def make_vignettes(n=1500, seed=7, atypical=0.08):
    """n synthetic cases. A share of 'atypical' cases (default 8%) present with the
    pattern of another condition, mimicking real diagnostic overlap and imperfect
    reference labels, so the classifier cannot be perfect."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(CLASSES, size=n, p=[0.36, 0.30, 0.34])
    notes = []
    for l in labels:
        shown = rng.choice([c for c in CLASSES if c != l]) if rng.random() < atypical else l
        notes.append(_note(rng, shown))
    return pd.DataFrame({"note": notes, "label": labels})
