"""Trial table for the Bayesian-observer model of the estimation reports.

One row per estimation-task trial (39 included participants x 2 sessions x 8 runs x
30 trials): participant, session, run, trial, range condition, presented n, the slider's
start position, the report, whether it was confirmed by a click, and the slider
trajectory summary needed to censor unfinished reports.

Response mechanics (experiment/task.py, stimuli.py, settings/scanner.yml):
- After the array (0.6 s) and a 4-6 s jitter, a horizontal slider appears for 3.0 s.
  It spans the run's range (narrow 10-25, wide 10-40; integer positions); its length
  scales with the range (3.75 vs 7.5 deg), so one step is the same distance in both.
- The marker is hidden until the participant moves the (MR-compatible) mouse; it then
  follows the mouse (int-truncated position, updated whenever the mouse moved more than
  one step since the last update) and its number is shown below the slider. The start
  marker position (uniform over the range) is where the mouse is placed, so the first
  visible position is the start +- 1.
- A left click confirms the response. Without a click the trial times out: 'Too late!'
  is shown and the events file has no response (NaN).
- The PsychoPy log has a line per frame with the displayed number while the marker is
  visible ("unnamed TextStim: text = 'k'"), so the full 60-Hz trajectory, including the
  final position on timeout trials, can be read back from the log. Log time = events
  time + an offset estimated from the scanner pulses (events 'pulse' rows vs log
  'Keypress: 5').

Censoring (same rule as retinonumeral, adapted to a mouse-driven slider): a timeout
trial whose trajectory was monotonic (no reversal among position changes), moved away
from the start, and whose last position change happened in the last 0.5 s of the
window is an unfinished report: censor = +1 (moving up: intended >= final) or -1.
Timeout trials that were moved but are not unfinished keep their final position as an
observed report (as in retinonumeral, where unconfirmed positions at timeout were kept).
Timeout trials without any movement have no report and are dropped from the model.

Writes /data/ds-neuralpriors/derivatives/behaviour_model/behaviour_trials.tsv and a
per-participant summary to notes/data/behaviour_model/.

    ~/mambaforge/envs/bauer/bin/python -m neural_priors.cogmodels.make_trial_table
"""
import argparse
import os
import os.path as op
import re

import numpy as np
import pandas as pd
import yaml

REPO = op.abspath(op.join(op.dirname(__file__), '..', '..'))
RANGES = {'narrow': (10, 25), 'wide': (10, 40)}
TEXT_RE = re.compile(r"^([0-9.]+)\s+\tEXP \tunnamed TextStim: text = '(\d+)'")
PULSE_RE = re.compile(r"^([0-9.]+)\s+\tDATA \tKeypress: 5")


def subjects():
    with open(op.join(REPO, 'neural_priors', 'data', 'subjects.yml')) as f:
        m = yaml.safe_load(f)
    return [k for k, v in m.items() if len(v) > 1], m


def parse_run(bids, sub, ses, run):
    stem = op.join(bids, 'sourcedata', 'behavior', f'sub-{sub}', f'ses-{ses}',
                   f'sub-{sub}_ses-{ses}_task-estimation_task_run-{run}')
    ev = pd.read_csv(stem + '_events.tsv', sep='\t')
    with open(stem + '_expsettings.yml') as f:
        st = yaml.safe_load(f)
    window_set = float(st['durations']['response_screen'])
    lines = open(stem + '_log.txt', errors='replace').read().splitlines()
    texts, pulses = [], []
    for ln in lines:
        m = TEXT_RE.match(ln)
        if m:
            texts.append((float(m.group(1)), int(m.group(2))))
            continue
        m = PULSE_RE.match(ln)
        if m:
            pulses.append(float(m.group(1)))
    texts = np.array(texts) if texts else np.zeros((0, 2))
    ev_pulses = ev.loc[ev.event_type == 'pulse', 'onset'].to_numpy()
    pulses = np.array(pulses)
    # offset: the candidate (log pulse - events pulse, first 10 x 10) that aligns the
    # most pulses within 20 ms; refined as the median over the aligned pairs
    best = (-1, 0.0)
    for a in pulses[:10]:
        for b in ev_pulses[:10]:
            o = a - b
            j = np.searchsorted(pulses, ev_pulses + o).clip(1, len(pulses) - 1)
            dd = np.minimum(np.abs(pulses[j] - ev_pulses - o), np.abs(pulses[j - 1] - ev_pulses - o))
            if (dd < .02).sum() > best[0]:
                best = ((dd < .02).sum(), o)
    o = best[1]
    j = np.searchsorted(pulses, ev_pulses + o).clip(1, len(pulses) - 1)
    near = np.where(np.abs(pulses[j] - ev_pulses - o) < np.abs(pulses[j - 1] - ev_pulses - o),
                    pulses[j], pulses[j - 1])
    ok = np.abs(near - ev_pulses - o) < .02
    off = np.median(near[ok] - ev_pulses[ok])
    off_spread = 1 - ok.mean()          # fraction of events pulses without a log match

    rng = 'wide' if (ev['n'].dropna() > 25).any() else 'narrow'
    lo, hi = RANGES[rng]
    # phase-onset rows of the response phase (key presses are also logged as event_type
    # 'response', but with the key in the response column)
    resp = ev[(ev.event_type == 'response') & ev.response.isna() & (ev.trial_nr > 0)
              & ev.n.notna()].drop_duplicates('trial_nr').set_index('trial_nr')
    fb = ev[(ev.event_type == 'feedback') & (ev.trial_nr > 0)].drop_duplicates(
        'trial_nr').set_index('trial_nr')
    rows = []
    for t in resp.index:
        r0 = resp.loc[t, 'onset'] + off
        r1 = (fb.loc[t, 'onset'] + off) if t in fb.index else r0 + window_set
        sel = (texts[:, 0] >= r0 - 1e-3) & (texts[:, 0] < r1 - 1e-3) if len(texts) else []
        tr = texts[sel] if len(texts) else texts
        start = int(resp.loc[t, 'start_marker_position'])
        response = pd.to_numeric(fb.loc[t, 'response'], errors='coerce') if t in fb.index else np.nan
        confirmed = bool(np.isfinite(response))
        row = dict(subject=sub, session=ses, run=run, trial_nr=int(t), range=rng,
                   range_lo=lo, range_hi=hi, n=int(resp.loc[t, 'n']), slider_start=start,
                   response_logged=response, confirmed=confirmed,
                   rt=pd.to_numeric(fb.loc[t, 'response_time'], errors='coerce') if confirmed else np.nan,
                   window=r1 - r0, n_frames_shown=len(tr), log_offset_spread=off_spread)
        if len(tr) == 0:
            row.update(moved=False, final=np.nan, reversals=0, net=0, t_last_change=np.nan,
                       t_first_shown=np.nan, path_len=0)
            rows.append(row)
            continue
        vals = tr[:, 1].astype(int)
        path = np.concatenate([[start], vals])
        steps = np.diff(path)
        nz = steps[steps != 0]
        rev = int(np.sum(np.diff(np.sign(nz)) != 0)) if len(nz) > 1 else 0
        ch = np.nonzero(np.diff(vals) != 0)[0]
        t_last = (tr[ch[-1] + 1, 0] - r0) if len(ch) else (tr[0, 0] - r0)
        row.update(moved=True, final=int(vals[-1]), reversals=rev, net=int(vals[-1] - start),
                   t_last_change=t_last, t_first_shown=tr[0, 0] - r0,
                   path_len=int(np.abs(nz).sum()))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or op.join(args.bids_folder, 'derivatives', 'behaviour_model',
                              'behaviour_trials.tsv')
    subs, mapping = subjects()
    rows = []
    for s in subs:
        for ses in mapping[s]:
            for run in range(1, 9):
                try:
                    rows += parse_run(args.bids_folder, s, ses, run)
                except FileNotFoundError as e:
                    print('missing', s, ses, run, e)
    d = pd.DataFrame(rows)

    # parser check: on confirmed trials the last displayed number must be the logged report
    conf = d[d.confirmed]
    mism = conf[conf.final != conf.response_logged]
    print(f'confirmed trials: {len(conf)}; final shown != logged response: {len(mism)} '
          f'({len(mism) / len(conf):.2%}); |diff|>1: {(np.abs(mism.final - mism.response_logged) > 1).sum()}; '
          f'no frames shown: {conf.final.isna().sum()}')
    print('fraction of unmatched pulses per run: max', d.log_offset_spread.max().round(4))

    d['monotonic'] = d.reversals == 0
    d['censor'] = 0
    unfinished = ((~d.confirmed) & d.moved & d.monotonic & (d.net != 0)
                  & ((d.window - d.t_last_change) < 0.5))
    d.loc[unfinished, 'censor'] = np.sign(d.loc[unfinished, 'net']).astype(int)
    # the model's report: logged response when confirmed, final position on timeout
    d['response'] = np.where(d.confirmed, d.response_logged, d.final)
    d['status'] = np.select([d.confirmed, d.moved & (d.censor != 0), d.moved],
                            ['confirmed', 'timeout_censored', 'timeout_observed'],
                            'timeout_no_movement')
    os.makedirs(op.dirname(out), exist_ok=True)
    d.to_csv(out, sep='\t', index=False)
    print('wrote', out, len(d))

    print(d.groupby('range').status.value_counts().unstack().to_string())
    S = d.groupby(['subject', 'range']).agg(
        trials=('n', 'size'), confirmed=('confirmed', 'mean'),
        timeout_moved=('status', lambda x: x.isin(['timeout_censored', 'timeout_observed']).mean()),
        censored=('censor', lambda c: (c != 0).mean()),
        no_movement=('status', lambda x: (x == 'timeout_no_movement').mean()),
        median_rt=('rt', 'median')).reset_index()
    sd = op.join(REPO, 'notes', 'data', 'behaviour_model')
    os.makedirs(sd, exist_ok=True)
    S.to_csv(op.join(sd, 'trial_status_by_subject_summary.tsv'), sep='\t', index=False,
             float_format='%.4f')
    print(S.groupby('range')[['confirmed', 'timeout_moved', 'censored', 'no_movement',
                              'median_rt']].describe().T.round(3).to_string())


if __name__ == '__main__':
    main()
