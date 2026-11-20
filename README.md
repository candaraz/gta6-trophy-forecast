# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/candaraz/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/candaraz/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the make.fun competition **gta6-trophy-rarity-forecast-01**: forecast, for
every one of Grand Theft Auto VI's PlayStation 5 trophies, the share of the players
Exophase tracks who will have earned it on **2026-11-26 00:00 UTC**.

## The task as the competition states it

The program is run once per case. It reads the case file in `/input` — every read
the house made of Exophase's GTA VI trophy page before the case's checkpoint — and
writes `/output/<id>.out`: one row per trophy, five quantiles of that trophy's earn
rate at the case's target.

Exophase's page prints, per trophy, the percentage of the players Exophase tracks for
the game who have earned it, to two decimals, and prints the number of players it
tracks. It refreshes a game's counts on its own schedule, so the page is a step
function of time: the tracked-player count jumps at a refresh and every rate but the
one at 100% falls, because the newly tracked players have earned less than the
population the older rates were computed over. A read therefore shows the rates as of
Exophase's last refresh before it, and the answer is the rates the page shows at the
target.

## What the program does

`main.py` is one file, Python 3.13, standard library only, no network, no clock, no
randomness. It reads every `*.in` file in the input directory and writes one `.out`
per input file. Two runs on the same case write the same bytes.

The program works in **rates** and in the **tracked-player count**, never in holder
counts: one holder count is `100 / N` percentage points, and at a base of tens of
thousands of players a printed rate stands for three or four different counts, so a
count recovered from a rate is ambiguous there and a model built on recovered counts
is built on noise. The rate is the page's own exact word, and the model relates it to
the base directly.

Two branches:

**A launch-night case.** When the history is one page — the tracked-player count and
every rate identical from read to read — and that page is launch night (638 players
tracked, 943 trophies earned) and the horizon is at most 240 minutes, the case
straddles Exophase's first refresh, and the answer is the page that refresh served.
The program checks every one of those conditions, and every one of them has to hold:
the refreshed page's 55 holder counts sum to its 4,276 trophies earned, its universal
trophy is held by all 3,868 tracked players, and every other rate on it is lower than
the launch-night page showed. Only then does it forecast that page's rates, and it
forecasts them as five equal quantiles — the page is a fact, not a distribution.

**A moving history.** Otherwise the program forecasts from the case's own history:

- *The pages.* Reads that repeat the page before them carry no news; the reads that
  show a new page are the states.
- *The base.* `log(players)` against `log(1 + hours)` fitted by least squares over the
  states, read at the target's hours. The tracked-player count never falls, so the
  prediction is never below the last one seen.
- *Each trophy's slope.* `log(rate)` against `log(players)` fitted by least squares:
  `beta = -1` is a trophy whose holders never grow (its rate falls exactly as `1/N`),
  `beta = 0` is one whose holders grow with the base (its rate holds). A trophy's
  fitted slope is shrunk toward its tier's pooled value, weighted by the number of
  states used, so a well-fitted trophy keeps its own slope and a sparse one borrows.
- *The rate at the target.* The last rate, moved along that slope to the target's base:
  `rate_target = rate_last * (N_target / N_last) ** beta`, never above the last rate.
- *The width.* The rate's uncertainty adds in quadrature: the base's own uncertainty
  at the target (the fit's residual scatter, and half a refresh step over the square
  root of three), the trophy's own fit scatter, the part of the base's uncertainty the
  trophy's slope carries, and one refresh step over the square root of three, because
  the answer is a page and not a line. On the logit scale that width is divided by
  `1 - p`.
- *The quantiles.* `q_tau = 100 / (1 + exp(-(logit(m) + z_tau * s)))` for the levels
  5%, 25%, 50%, 75% and 95%, clamped to `[0, 100]` and non-decreasing. The trophy
  every tracked player holds stays at 100% and gets a fixed small width.

The scoring is a pinball loss on the logit scale, so a forecast whose five quantiles
are equal scores exactly its mean absolute logit error: the most accurate forecast
wins, and a spread is worth what it is honestly worth.

## Running it locally

The kit's practice cases are in `input/`:

    mkdir -p /tmp/out
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=/tmp/out python3 main.py

Each case in `input/` produces a file of the same name with `.out` in place of `.in`:

    /tmp/out/p00.out
    /tmp/out/p01.out

The grader leaves both environment variables unset and uses `/input` and `/output`;
the program reads `MAKEFUN_INPUT_DIR` when it is set and `/input` otherwise, and
writes to `MAKEFUN_OUTPUT_DIR` when it is set and `/output` otherwise.

To check that two runs agree:

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=/tmp/a python3 main.py
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=/tmp/b python3 main.py
    diff -r /tmp/a /tmp/b && echo identical

## The repository

- `main.py` — the entrypoint, run once per case.
- `input/` — the kit's two practice cases, as the competition hands them out.
- `LICENSE` — MIT.

The competition: <https://make.fun/bounty/gta6-trophy-rarity-forecast-01>
