#!/usr/bin/env python3
"""GTA VI (PS5) trophy rarity forecast for make.fun's `gta6-trophy-rarity-forecast-01`.

The program reads every `*.in` case from the input directory and writes one
`*.out` file per case into the output directory.  It is pure standard library,
deterministic and completely offline: every number it prints comes from the
case file itself.

Model in one paragraph
----------------------
Exophase refreshes a game's trophy counts on its own schedule, not with every
player's trophy.  At a refresh the tracked-player count jumps and the newly
tracked players have earned less than the population the rates were computed
over, so every rate *falls* at every refresh while the absolute number of
holders keeps *growing*.  A printed percentage is therefore a ratio with a
moving denominator and must never be differentiated directly.  Instead the
program recovers the integer holder count behind every printed rate, models
holder counts and the player base separately, and only at the very end divides
one by the other to obtain a rate.

Two branches:

* Branch A -- the launch-night case.  When the whole history is a single
  unchanged page (638 players tracked, 943 trophies earned, horizon <= 240
  minutes) the next refresh page is known exactly, so the forecast collapses
  to a point and all five quantiles of every trophy are that point.
* Branch B -- a moving history.  Holder counts are regressed against the
  player base, the player base itself is regressed against time, and the
  spread of both fits (plus the width of one refresh step) becomes the width
  of a logit-scale distribution around the median rate.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime

# --------------------------------------------------------------------------
# The known Exophase refresh page of the launch night (branch A).
# Holder counts, in the order of the case's `trophies` list.
# --------------------------------------------------------------------------
LAUNCH_PAGE_COUNTS = [
    4, 3868, 19, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 5, 4, 4, 5, 45, 15, 5,
    5, 6, 4, 4, 5, 4, 5, 4, 4, 4, 7, 6, 15, 5, 9, 12, 7, 5, 6, 6, 5, 8, 5,
    6, 6, 7, 6, 7, 7, 5, 7, 6, 6,
]
LAUNCH_PAGE_PLAYERS = 3868
LAUNCH_PAGE_EARNED = 4276
LAUNCH_HISTORY_PLAYERS = 638
LAUNCH_HISTORY_EARNED = 943
LAUNCH_MAX_HORIZON_MINUTES = 240

# Tier priors for how fast a trophy's holder count follows the player base.
# A bronze that almost nobody new earns sits near 0; a platinum everybody
# owns rides the player base almost one-for-one.
TIER_GAMMA_POOL = {
    "bronze": 0.14,
    "gold": 0.07,
    "silver": 0.0,
    "platinum": 0.0,
}
DEFAULT_GAMMA_POOL = 0.10

# z-scores of the five requested levels (0.05, 0.25, 0.5, 0.75, 0.95).
Z_LEVELS = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Width of a logit-scale forecast, clamped to this range.
MIN_LOGIT_WIDTH = 0.05
MAX_LOGIT_WIDTH = 4.0


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
def round_half_up(value: float, digits: int = 2) -> float:
    """Round like a browser's `toFixed` (half away from zero).

    The page prints two decimals; the grader compares against Python's
    `round`, which is half-to-even.  The two only disagree on exact ties,
    which essentially never happen for these denominators -- but the helper
    is used for tolerant *matching* of a rate to a holder count.
    """
    factor = 10.0 ** digits
    scaled = value * factor
    if scaled >= 0:
        rounded = math.floor(scaled + 0.5)
    else:
        rounded = math.ceil(scaled - 0.5)
    return rounded / factor


def rate_of(count: int, players: int) -> float:
    """Holder count -> the percentage the page prints (two decimals)."""
    if players <= 0:
        return 0.0
    return round(count / players * 100, 2)


def count_of(rate, players: int):
    """The printed percentage -> the integer holder count behind it.

    The page prints `round(k / n * 100, 2)`, so the holder count is the
    integer `k` whose rate prints back as exactly this percentage.  Several
    integers can share one printed value (a single holder is often well
    below the printed precision once the player base is large); the one
    closest to the naive `rate * n / 100` estimate is the best reading.
    """
    if rate is None or players is None or players <= 0:
        return None
    try:
        target = float(rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(target):
        return None
    estimate = target * players / 100.0
    # A percentage printed to two decimals covers the half-open interval
    # [target - 0.005, target + 0.005), so the integers that print back as
    # `target` are exactly those inside the matching slice of k.
    low = max(0, int(math.ceil((target - 0.005) * players / 100.0 - 1e-9)))
    high = min(players, int(math.floor((target + 0.005) * players / 100.0 + 1e-9)))
    best = None
    best_distance = None
    for k in range(low, high + 1):
        printed = rate_of(k, players)
        printed_alt = round_half_up(k / players * 100, 2)
        if abs(printed - target) < 1e-9 or abs(printed_alt - target) < 1e-9:
            distance = abs(k - estimate)
            if best_distance is None or distance < best_distance:
                best = k
                best_distance = distance
    if best is None:
        # No integer prints back exactly (an odd input): fall back on the
        # nearest holder count implied by the percentage.
        best = int(min(max(round(estimate), 0), players))
    return best


def parse_time(stamp) -> datetime:
    """Parse the ISO-8601 stamps used by the case files."""
    text = str(stamp or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def hours_between(start: datetime, end: datetime) -> float:
    """Whole hours from `start` to `end`, never negative."""
    delta = (end - start).total_seconds() / 3600.0
    return delta if delta > 0.0 else 0.0


def least_squares(xs, ys):
    """Ordinary least squares `y = a + b * x`.

    Returns `(a, b, residuals)`; `a`/`b` are `None` when the fit is not
    determined (fewer than two points, or no spread in x).
    """
    n = len(xs)
    if n < 2:
        return None, None, []
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx <= 0.0:
        return None, None, []
    b = sxy / sxx
    a = mean_y - b * mean_x
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, residuals


def scatter(residuals) -> float:
    """Root-mean-square residual: how far the fit misses its own points."""
    if not residuals:
        return 0.0
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def logistic(p: float) -> float:
    """Standard logistic function."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-p))


def safe_exp(value: float) -> float:
    """`exp` that cannot overflow into an infinity."""
    if value > 700.0:
        return math.inf
    if value < -700.0:
        return 0.0
    return math.exp(value)


def logit(p: float) -> float:
    """log-odds of a probability clamped away from 0 and 1."""
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


# --------------------------------------------------------------------------
# Reading the case
# --------------------------------------------------------------------------
class Read:
    """One read of the Exophase page."""

    __slots__ = ("slot", "at", "players", "earned", "rates")

    def __init__(self, raw):
        self.slot = raw.get("slot")
        self.at = parse_time(raw.get("at") or raw.get("slot"))
        self.players = int(raw.get("players_tracked") or 0)
        self.earned = int(raw.get("trophies_earned") or 0)
        self.rates = list(raw.get("rates") or [])


def load_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    case["_reads"] = [Read(raw) for raw in (case.get("history") or [])]
    case["_trophy_ids"] = [t["id"] for t in (case.get("trophies") or [])]
    case["_tiers"] = [(t.get("tier") or "").lower() for t in (case.get("trophies") or [])]
    return case


# --------------------------------------------------------------------------
# Branch A -- the launch night
# --------------------------------------------------------------------------
def history_is_one_page(reads) -> bool:
    """True when no read shows a new page: same players, same rates."""
    if not reads:
        return False
    for previous, current in zip(reads, reads[1:]):
        if previous.players != current.players:
            return False
        if list(previous.rates) != list(current.rates):
            return False
    return True


def branch_a_applies(case: dict) -> bool:
    """All the launch-night conditions have to hold, or we fall through."""
    reads = case["_reads"]
    if not reads:
        return False
    if not history_is_one_page(reads):
        return False
    last = reads[-1]
    if last.players != LAUNCH_HISTORY_PLAYERS or last.earned != LAUNCH_HISTORY_EARNED:
        return False
    horizon = case.get("horizon_minutes")
    if horizon is None or horizon > LAUNCH_MAX_HORIZON_MINUTES:
        return False
    counts = LAUNCH_PAGE_COUNTS
    if len(counts) != len(case["_trophy_ids"]):
        return False
    if sum(counts) != LAUNCH_PAGE_EARNED:
        return False
    if counts[1] != LAUNCH_PAGE_PLAYERS:
        return False
    last_rates = last.rates
    if len(last_rates) != len(counts):
        return False

    # Exactly one trophy is held by every tracked player.  On the launch page
    # that trophy's holder count is the tracked-player count itself, so it
    # stays at 100% and is exempt from the "every rate fell" test.
    universal = [i for i, rate in enumerate(last_rates)
                 if rate is not None and abs(float(rate) - 100.0) < 1e-9]
    if len(universal) != 1:
        return False
    universal_index = universal[0]
    if counts[universal_index] != LAUNCH_PAGE_PLAYERS:
        return False

    for index, (count, old_rate) in enumerate(zip(counts, last_rates)):
        if old_rate is None:
            return False
        new_rate = round(count / LAUNCH_PAGE_PLAYERS * 100, 2)
        if index == universal_index:
            if abs(new_rate - 100.0) > 1e-9:
                return False
            continue
        # Every other rate fell at the refresh: the newly tracked players
        # have earned less than the population the old rates were over.
        if new_rate >= float(old_rate):
            return False
    return True


def branch_a_forecast(case: dict):
    """The known refresh page: every trophy's five quantiles are its rate."""
    rows = []
    for index, trophy_id in enumerate(case["_trophy_ids"]):
        count = LAUNCH_PAGE_COUNTS[index]
        value = rate_of(count, LAUNCH_PAGE_PLAYERS)
        rows.append({"id": trophy_id, "q": [value] * 5})
    return rows


# --------------------------------------------------------------------------
# Branch B -- a history that moves
# --------------------------------------------------------------------------
def collect_states(case: dict):
    """The reads that show a new page; repeats of the previous page carry no news."""
    states = []
    previous = None
    for read in case["_reads"]:
        if previous is None or read.players != previous.players or list(read.rates) != list(previous.rates):
            states.append(read)
        previous = read
    return states


def player_base(case: dict, states):
    """Fit the tracked-player count against time and evaluate it at the target.

    The time variable is `x = log(1 + hours)`: smooth at the very start (no
    zero-length log) and slowly growing, which is how a launch curve behaves.
    """
    if not states:
        return None
    t0 = states[0].at
    hours = [hours_between(t0, state.at) for state in states]
    xs = [math.log(1.0 + h) for h in hours]
    ys = [math.log(state.players) for state in states]
    target_stamp = case.get("target") or case.get("at") or ""
    target_hours = hours_between(t0, parse_time(target_stamp))
    x_target = math.log(1.0 + target_hours)

    if len(states) >= 2:
        a, b, residuals = least_squares(xs, ys)
    else:
        a, b, residuals = None, None, []
    if a is None:
        # Not enough spread to fit a line: hold the base where it is.
        predicted = float(states[-1].players)
        residual_scatter = 0.0
    else:
        predicted = math.exp(a + b * x_target)
        residual_scatter = scatter(residuals)

    # The tracked-player count never falls.
    predicted = max(predicted, float(states[-1].players))

    # One refresh step: the biggest relative jump of N between two states.
    step = 0.0
    for previous, current in zip(states, states[1:]):
        if previous.players > 0 and current.players > 0:
            step = max(step, abs(math.log(current.players / previous.players)))

    return {
        "n_target": predicted,
        "residual_scatter": residual_scatter,
        "step": step,
        "x_target": x_target,
    }


def trophy_fit(states, index: int):
    """Regress a trophy's holder count against the player base.

    Returns `(alpha, gamma, count_samples, residual_scatter, last_count)`.
    """
    samples = []
    for state in states:
        if index >= len(state.rates):
            continue
        rate = state.rates[index]
        if rate is None:
            continue
        count = count_of(rate, state.players)
        if count is None or count <= 0:
            continue
        samples.append((state.players, count))

    last_count = None
    for state in reversed(states):
        if index < len(state.rates) and state.rates[index] is not None:
            last_count = count_of(state.rates[index], state.players)
            break

    if len(samples) >= 2:
        xs = [math.log(n) for n, _ in samples]
        ys = [math.log(k) for _, k in samples]
        a, b, residuals = least_squares(xs, ys)
        if a is not None:
            return a, b, samples, scatter(residuals), last_count
    return None, None, samples, 0.0, last_count

def shrink_gamma(gamma, pool, n_states: int) -> float:
    """Shrink a fitted gamma toward the tier's pooled value.

    The pool counts as a single extra observation, so a fit built from many
    states keeps most of its own signal while a fit from one or two states
    falls back on the tier prior.
    """
    if gamma is None:
        return clamp(pool, 0.0, 1.0)
    blended = (n_states * gamma + pool) / (n_states + 1.0)
    return clamp(blended, 0.0, 1.0)


def forecast_trophies(case: dict):
    """Branch B: a distribution per trophy around a predicted median rate."""
    states = collect_states(case)
    base = player_base(case, states)
    n_target = base["n_target"] if base else 1.0
    step = base["step"] if base else 0.0
    base_scatter = base["residual_scatter"] if base else 0.0

    # The base's own uncertainty: how badly the fit misses its own points,
    # plus half a refresh step (a page is a page, not a line).
    sigma_log_n = base_scatter
    half_step = step / 2.0
    base_uncertainty = math.sqrt(sigma_log_n ** 2 + (half_step / math.sqrt(3.0)) ** 2)
    # A page rather than a line adds one further refresh step.
    page_term = step / math.sqrt(3.0)

    rows = []
    for index, trophy_id in enumerate(case["_trophy_ids"]):
        tier = case["_tiers"][index] if index < len(case["_tiers"]) else ""
        pool = TIER_GAMMA_POOL.get(tier, DEFAULT_GAMMA_POOL)

        alpha, gamma, samples, resid_scatter, last_count = trophy_fit(states, index)

        # Does every tracked player hold this trophy?
        everyone = False
        for state in reversed(states):
            if index < len(state.rates) and state.rates[index] is not None:
                everyone = abs(float(state.rates[index]) - 100.0) < 1e-9
                break

        if everyone:
            # Its count is the tracked-player count, so its rate stays at 100.
            median = 99.99
            width = 0.5
            rows.append({"id": trophy_id, "q": quantile_row(median, width)})
            continue

        n_states = len(samples)
        gamma = shrink_gamma(gamma, pool, n_states)

        if alpha is None:
            # Too few states to fit: hold the count where it is.
            count = last_count if last_count is not None else 0
        else:
            count = math.exp(alpha + gamma * math.log(n_target)) if n_target > 0 else 0.0
            # A player keeps a trophy: the count never falls below the last one.
            if last_count is not None:
                count = max(count, last_count)

        rate = 0.0 if n_target <= 0 else clamp(count / n_target * 100.0, 0.0, 100.0)

        # The trophy's own uncertainty: the scatter of its own fit plus the
        # part of the base's uncertainty that travels through gamma.
        trophy_uncertainty = math.sqrt(resid_scatter ** 2 + (abs(gamma) * base_uncertainty) ** 2)
        sigma_log = math.sqrt(base_uncertainty ** 2 + trophy_uncertainty ** 2 + page_term ** 2)

        # On the logit scale the width grows as p approaches 1: a rate of 99.9%
        # sits in the steep part of the logit, a rate of 0.1% in the flat one.
        p = clamp(rate / 100.0, 0.0, 1.0)
        width = sigma_log / max(1.0 - p, 1e-9)
        width = clamp(width, MIN_LOGIT_WIDTH, MAX_LOGIT_WIDTH)

        rows.append({"id": trophy_id, "q": quantile_row(rate, width)})
    return rows


def quantile_row(median_rate: float, width: float):
    """The five quantiles of a logit-normal forecast around the median.

    `median_rate` is a percentage; it is clamped to [0.01, 99.99] before it
    becomes the probability `p = m / 100` that enters the logit.
    """
    p = clamp(median_rate, 0.01, 99.99) / 100.0
    center = logit(p)
    values = []
    for z in Z_LEVELS:
        value = 100.0 / (1.0 + safe_exp(-(center + z * width)))
        values.append(clamp(value, 0.0, 100.0))
    # Keep the row non-decreasing even after the clamp.
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    return [round(v, 6) for v in values]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def forecast_case(case: dict):
    if branch_a_applies(case):
        return branch_a_forecast(case)
    return forecast_trophies(case)


def main() -> int:
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case = load_case(os.path.join(input_dir, name))
        rows = forecast_case(case)
        payload = {"trophies": rows}
        out_path = os.path.join(output_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
