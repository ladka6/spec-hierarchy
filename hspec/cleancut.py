"""Score speculative decoding only on the clean part of an output.

Greedy base models in Q/A format either start a new "Question:" turn or fall into loops. Both are easy for any
drafter and inflate tau. Greedy decoding gives (up to numerics) the same output for every method, so cutting each
output at the first new turn or loop onset and counting only the rounds before the cut compares all methods on
the same text.
"""

from __future__ import annotations


def loop_onset(t, n=8, win=64, thr=0.75, lookback=256):
    """First position where repetition becomes sustained: over the next `win` positions more than `thr` of the
    n-grams already occurred within the previous `lookback` tokens. None if the text never loops."""
    last, rep = {}, []
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        j = last.get(g)
        rep.append(j is not None and i - j <= lookback)
        last[g] = i
    if len(rep) < win:
        return None
    s = sum(rep[:win])
    for i in range(len(rep) - win + 1):
        if i:
            s += rep[i + win - 1] - rep[i - 1]
        if s > thr * win:
            return i
    return None


_piece: dict = {}


def turn_onset(tok, t, marker="Question"):
    """Index just after the newline that starts a new `marker` turn, or None."""
    def piece(x):
        if x not in _piece:
            _piece[x] = tok.decode([x])
        return _piece[x]
    for i in range(len(t) - 1):
        if piece(t[i]).endswith("\n") and piece(t[i + 1]).lstrip().startswith(marker):
            return i + 1
    return None


def clean_len(tok, t) -> int:
    cuts = [c for c in (turn_onset(tok, t), loop_onset(t)) if c is not None]
    return min(cuts) if cuts else len(t)


def clean_stats(tok, r) -> dict:
    """tau, tokens and (round-proportional) decode time over the clean prefix of a GenResult."""
    t = r.generated.tolist()
    c = clean_len(tok, t)
    k, cum = 0, 0
    for n in r.rounds:
        if cum >= c:
            break
        cum += n
        k += 1
    k = max(k, 1)
    frac = k / max(len(r.rounds), 1)
    return {"tokens_clean": c, "rounds_clean": k, "tau_clean": c / k, "clean_frac": c / max(len(t), 1),
            "ms_clean": r.decode_time * 1000 * frac}
