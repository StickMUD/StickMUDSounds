#!/usr/bin/env python3
"""
Generate the StickMUD combat sound set.

Design goals (see the corresponding design discussion):

  * Low pitch -> something happened to a body. High pitch -> info / cue.
  * "Self" sounds carry more sub-500 Hz energy than their "other" twin,
    so the same event reads as heavier when it lands on the player.
  * Crit deliberately breaks the bass bed: it sits an octave above the
    hit sounds so it pokes out of a stream of thuds.
  * Duration ladders the consequence: miss < hit/crit < death.
  * Loudness stays in a narrow band; importance is signalled by timbre
    and length, not by being loud.

Output: 44.1 kHz, stereo, 128 kbps MP3 in StickMUDSounds/sounds/combat/.
Existing files are backed up to *.mp3.bak the first time the script runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


SR = 44100
OUT_DIR = Path(__file__).resolve().parent.parent / "sounds" / "combat"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def t_arr(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR), dtype=np.float64) / SR


def sweep_sine(start_hz: float, end_hz: float, seconds: float) -> np.ndarray:
    """Linear-frequency sine sweep from start_hz to end_hz over `seconds`."""
    t = t_arr(seconds)
    if len(t) == 0:
        return t
    # Instantaneous frequency goes start -> end linearly; integrate to phase.
    freqs = np.linspace(start_hz, end_hz, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    return np.sin(phase)


def exp_decay(seconds: float, tau: float) -> np.ndarray:
    t = t_arr(seconds)
    return np.exp(-t / tau)


def fade_edges(x: np.ndarray, fade_in_ms: float = 3.0, fade_out_ms: float = 12.0) -> np.ndarray:
    """Smooth the very edges of a clip to avoid click artefacts."""
    n = len(x)
    fi = max(1, int(SR * fade_in_ms / 1000.0))
    fo = max(1, int(SR * fade_out_ms / 1000.0))
    fi = min(fi, n // 2)
    fo = min(fo, n // 2)
    env = np.ones(n)
    env[:fi] = np.linspace(0.0, 1.0, fi)
    env[-fo:] = np.linspace(1.0, 0.0, fo)
    return x * env


def one_pole_lp(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Cheap one-pole lowpass; good enough for shaping noise/transients."""
    if cutoff_hz >= SR / 2:
        return x.copy()
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    a = (1.0 / SR) / (rc + 1.0 / SR)
    y = np.empty_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = prev + a * (v - prev)
        y[i] = prev
    return y


def one_pole_hp(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return x.copy()
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    a = rc / (rc + 1.0 / SR)
    y = np.empty_like(x)
    prev_x = 0.0
    prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return y


def bandpass_noise(seconds: float, lo_hz: float, hi_hz: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    if n == 0:
        return np.zeros(0)
    noise = rng.standard_normal(n)
    return one_pole_lp(one_pole_hp(noise, lo_hz), hi_hz)


def swept_bandpass_noise(seconds: float,
                         lo_start: float, lo_end: float,
                         hi_start: float, hi_end: float,
                         seed: int = 0) -> np.ndarray:
    """Block-wise swept bandpass noise (cheap and effective for whooshes)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    if n == 0:
        return np.zeros(0)
    block = 64
    out = np.zeros(n)
    noise = rng.standard_normal(n)
    los = np.linspace(lo_start, lo_end, max(1, n // block + 1))
    his = np.linspace(hi_start, hi_end, max(1, n // block + 1))
    pos = 0
    for k in range(len(los)):
        end = min(pos + block, n)
        if pos >= n:
            break
        seg = noise[pos:end]
        seg = one_pole_hp(seg, los[k])
        seg = one_pole_lp(seg, his[k])
        out[pos:end] = seg
        pos = end
    return out


def peak_normalize(x: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return x
    target = 10 ** (target_dbfs / 20.0)
    return x * (target / peak)


# ---------------------------------------------------------------------------
# Event recipes
# ---------------------------------------------------------------------------

def make_hit_self() -> np.ndarray:
    """You deal damage. Drier, lighter thwack -- spammy, must not fatigue."""
    seconds = 0.13
    body = sweep_sine(280, 220, seconds) * exp_decay(seconds, tau=0.040)
    harm = sweep_sine(560, 440, seconds) * exp_decay(seconds, tau=0.025) * 0.35
    tick = bandpass_noise(0.012, 700.0, 1500.0, seed=22)
    tick = tick * np.linspace(1.0, 0.0, len(tick)) * 0.25
    out = body + harm
    out[:len(tick)] += tick
    out = one_pole_hp(out, 90.0)
    out = one_pole_lp(out, 1800.0)
    return fade_edges(out, 2, 14)


def make_hit_other() -> np.ndarray:
    """You take damage. Deep body thud -- the event the player must notice."""
    seconds = 0.18
    body = sweep_sine(180, 90, seconds) * exp_decay(seconds, tau=0.060)
    sub = sweep_sine(90, 55, seconds) * exp_decay(seconds, tau=0.090) * 0.7
    click_len = 0.004
    click = (np.random.default_rng(11).standard_normal(int(click_len * SR))
             * np.linspace(1.0, 0.0, int(click_len * SR)))
    click = one_pole_lp(click, 600.0) * 0.35
    out = body + sub
    out[:len(click)] += click
    out = one_pole_lp(out, 350.0)
    return fade_edges(out, 2, 18)


def make_miss_self() -> np.ndarray:
    """Sharp whoosh: your blade cut air. Mid band, no low end, no shrill top."""
    seconds = 0.10
    n = int(seconds * SR)
    body = swept_bandpass_noise(seconds, 700, 900, 1300, 1500, seed=33)
    # Triangle-ish envelope: fast in, slower out.
    env = np.concatenate([
        np.linspace(0.0, 1.0, int(0.030 * SR)),
        np.linspace(1.0, 0.0, n - int(0.030 * SR)),
    ])
    out = body * env
    out = one_pole_hp(out, 250.0)
    out = one_pole_lp(one_pole_lp(out, 1800.0), 1800.0)
    return fade_edges(out, 2, 8)


def make_miss_other() -> np.ndarray:
    """Softer whoosh you hear nearby. Sits between miss_self and hit_other."""
    seconds = 0.11
    n = int(seconds * SR)
    body = swept_bandpass_noise(seconds, 400, 500, 800, 1000, seed=44)
    env = np.concatenate([
        np.linspace(0.0, 1.0, int(0.035 * SR)),
        np.linspace(1.0, 0.0, n - int(0.035 * SR)),
    ])
    out = body * env * 0.85
    out = one_pole_hp(out, 200.0)
    out = one_pole_lp(one_pole_lp(out, 1200.0), 1200.0)
    return fade_edges(out, 2, 10)


def make_crit() -> np.ndarray:
    """Bell-like sting an octave above the hit sounds."""
    seconds = 0.22
    f1, f2, f3 = 1320.0, 2640.0, 1980.0  # E6, E7, slightly inharmonic shimmer
    t = t_arr(seconds)
    bell = (np.sin(2 * np.pi * f1 * t) * np.exp(-t / 0.130)
            + 0.5 * np.sin(2 * np.pi * f2 * t) * np.exp(-t / 0.090)
            + 0.25 * np.sin(2 * np.pi * f3 * t) * np.exp(-t / 0.110))
    # Sharp pluck transient.
    pluck = bandpass_noise(0.006, 1500.0, 4000.0, seed=55)
    pluck = pluck * np.linspace(1.0, 0.0, len(pluck)) * 0.3
    out = bell
    out[:len(pluck)] += pluck
    return fade_edges(out, 1, 25)


def make_death_self() -> np.ndarray:
    """Long, heavy sub-bass groan. The deepest and the longest in the set."""
    seconds = 0.50
    fund = sweep_sine(75, 40, seconds) * exp_decay(seconds, tau=0.250)
    second = sweep_sine(120, 80, seconds) * exp_decay(seconds, tau=0.180) * 0.35
    # Soft rumble bed.
    rumble = bandpass_noise(seconds, 30.0, 120.0, seed=66) * 0.15
    rumble = rumble * exp_decay(seconds, tau=0.300)
    out = fund + second + rumble
    out = one_pole_lp(out, 250.0)
    # Slow attack so it swells in rather than punching.
    n = len(out)
    attack = int(0.030 * SR)
    env = np.ones(n)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    out = out * env
    return fade_edges(out, 5, 60)


def make_death_other() -> np.ndarray:
    """Enemy dies: a brighter, falling triumph cue with a soft ring-out.

    Distinct from death_self -- this is a satisfying punctuation that the
    player has won the exchange, not a catastrophe befalling them.
    """
    seconds = 0.35
    n = int(seconds * SR)
    # Descending tone: a body falling. Starts mid, lands low.
    drop = sweep_sine(330, 120, seconds) * exp_decay(seconds, tau=0.140)
    # Sympathetic upper partial that fades faster -- gives it the "ring".
    ring = sweep_sine(660, 240, seconds) * exp_decay(seconds, tau=0.080) * 0.35
    # Brief impact at the moment of the drop.
    thud = bandpass_noise(0.020, 200.0, 900.0, seed=77)
    thud = thud * np.linspace(1.0, 0.0, len(thud)) * 0.4
    out = drop + ring
    out[:len(thud)] += thud
    out = one_pole_hp(out, 80.0)
    out = one_pole_lp(out, 1600.0)
    # Gentle swell-in so the descent reads as a fall rather than a punch.
    attack = int(0.010 * SR)
    env = np.ones(n)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    out = out * env
    return fade_edges(out, 3, 35)


RECIPES = {
    "hit_self.mp3": make_hit_self,
    "hit_other.mp3": make_hit_other,
    "miss_self.mp3": make_miss_self,
    "miss_other.mp3": make_miss_other,
    "crit.mp3": make_crit,
    "death_self.mp3": make_death_self,
    "death_other.mp3": make_death_other,
}


# ---------------------------------------------------------------------------
# WAV / MP3 plumbing
# ---------------------------------------------------------------------------

def write_wav(path: Path, mono: np.ndarray) -> None:
    """Write a 16-bit stereo WAV (duplicate the mono channel)."""
    mono = peak_normalize(mono, target_dbfs=-3.0)
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    stereo = np.stack([pcm, pcm], axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def wav_to_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(wav),
            "-codec:a", "libmp3lame",
            "-b:a", "128k",
            str(mp3),
        ],
        check=True,
    )


def backup_originals(out_dir: Path) -> None:
    for mp3 in out_dir.glob("*.mp3"):
        bak = mp3.with_suffix(".mp3.bak")
        if not bak.exists():
            shutil.copy2(mp3, bak)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_originals(out_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, recipe in RECIPES.items():
            wav = tmp / (name.replace(".mp3", ".wav"))
            audio = recipe()
            write_wav(wav, audio)
            mp3 = out_dir / name
            wav_to_mp3(wav, mp3)
            print(f"wrote {mp3.relative_to(out_dir.parent.parent)} "
                  f"({mp3.stat().st_size} bytes, {len(audio)/SR:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
