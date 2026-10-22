"""Export MIDI : transcription des pistes Demucs (basic-pitch + détection d'attaques batterie). Lancé par midi.sh."""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import librosa, mido, soundfile as sf

ROOT = Path(__file__).resolve().parent
SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
PPQ = 480
SR = 22050
DRUM = {"kick": 36, "snare": 38, "hat": 42, "crash": 49}

# programmes GM : famille d'instruments AudioSet (piste "other") -> (programme, labels)
FAMILIES = {
    "piano": (0, ["Piano", "Electric piano", "Keyboard (musical)"]),
    "guitare": (None, ["Guitar", "Plucked string instrument", "Electric guitar", "Acoustic guitar", "Strum"]),
    "cordes": (48, ["Violin, fiddle", "Bowed string instrument", "String section", "Orchestra", "Cello"]),
    "orgue": (16, ["Organ", "Electronic organ", "Hammond organ"]),
}


# ---------- tempo / mesure ----------
def analyse_tempo(drums, mix):
    """Retourne dict(meter '4/4'|'12/8', period (s par temps), phase (s), downbeat_k, details)."""
    env = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=256, aggregate=np.median)
    env_mix = librosa.onset.onset_strength(y=mix, sr=SR, hop_length=256)
    env = env / (env.max() + 1e-9) + 0.5 * env_mix / (env_mix.max() + 1e-9)
    fps = SR / 256
    ac = librosa.autocorrelate(env - env.mean(), max_size=int(fps * 4))
    ac = ac / (ac[0] + 1e-9)

    def at(sec):
        i = sec * fps
        lo = int(np.floor(i)); f = i - lo
        return (1 - f) * ac[lo] + f * ac[min(lo + 1, len(ac) - 1)]

    best = {}
    for name, divs, trange in (("4/4", (1, 2, 4), (0.46, 0.92)), ("12/8", (1, 3, 6), (0.70, 1.50))):
        for T in np.arange(*trange, 0.01):
            s = sum(at(T / d) * (1.0 if d == 1 else 0.7) for d in divs) + 0.5 * at(2 * T)
            if name not in best or s > best[name][0]:
                best[name] = (s, T)
    meter = "12/8" if best["12/8"][0] > 1.5 * best["4/4"][0] else "4/4"  # 12/8 seulement si net avantage
    T0 = best[meter][1]
    bpm0 = 60 / T0
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=256, start_bpm=bpm0, tightness=200, units="time")
    beats = np.asarray(beats)
    k = np.arange(len(beats))
    from scipy.ndimage import uniform_filter1d
    med = np.median(np.diff(beats))
    good = np.abs(np.diff(beats) / med - 1) < 0.12   # intervalles proches du tempo médian
    first = int(np.argmax(good)) if good.any() else 0
    last = len(good) - int(np.argmax(good[::-1])) if good.any() else len(beats)
    beats = beats[first:last + 1]                    # écarte les temps instables en début/fin
    smooth = uniform_filter1d(beats, size=5, mode="nearest")  # lisse la gigue du beat tracking
    period = float(np.median(np.diff(smooth)))
    bpm_loc = 60 / np.diff(smooth)
    # downbeat : parmi 4 phases, celle où l'énergie grave de la batterie est maximale
    S = np.abs(librosa.stft(drums, n_fft=2048, hop_length=256))
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = S[(fr >= 30) & (fr <= 120)].sum(0)
    idx = np.clip((beats * fps).astype(int), 0, len(low) - 1)
    scores = [low[idx[p::4]].mean() if len(idx[p::4]) else 0 for p in range(4)]
    return {"meter": meter, "period": period, "beats": smooth, "downbeat": int(np.argmax(scores)),
            "scores": {"4/4": round(best["4/4"][0], 3), "12/8": round(best["12/8"][0], 3)},
            "n_beats": int(len(beats)), "dur": float(len(drums) / SR),
            "debut_grille_s": round(float(beats[0]), 2), "bpm_p5": round(float(np.percentile(bpm_loc, 5)), 1), "bpm_p95": round(float(np.percentile(bpm_loc, 95)), 1)}


# ---------- batterie ----------
def band_flux(S, fr, lo, hi):
    b = np.log1p(S[(fr >= lo) & (fr <= hi)] * 30).sum(0)
    return np.maximum(np.diff(b, prepend=b[0]), 0)


def pick(flux, fps, k=1.6, min_gap=0.06, floor=None):
    med = np.convolve(flux, np.ones(int(fps)) / int(fps), mode="same")  # moyenne glissante 1 s
    thr = med * k + (floor if floor is not None else flux.std() * 0.5)
    pk = librosa.util.peak_pick(flux, pre_max=int(0.03 * fps), post_max=int(0.03 * fps), pre_avg=int(0.1 * fps),
                                post_avg=int(0.1 * fps), delta=0, wait=int(min_gap * fps))
    return [p for p in pk if flux[p] > thr[p]]


def transcribe_drums(drums):
    hop = 256; fps = SR / hop
    S = np.abs(librosa.stft(drums, n_fft=2048, hop_length=hop))
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    fk, fs_, fh = band_flux(S, fr, 30, 130), band_flux(S, fr, 1500, 5000), band_flux(S, fr, 7000, 10500)
    mid = S[(fr >= 180) & (fr <= 350)].sum(0)
    hi_env = S[(fr >= 7000) & (fr <= 10500)].sum(0)
    ev = []
    ks = pick(fk, fps, 1.5, 0.08)
    for p in ks:
        ev.append((p / fps, "kick", fk[p] / (fk.max() + 1e-9)))
    kick_t = np.array([e[0] for e in ev]) if ev else np.array([])
    snares = []
    for p in pick(fs_, fps, 1.5, 0.1):
        if mid[p:p + 4].max() > np.median(mid) * 1.8:  # corps de caisse claire
            snares.append(p); ev.append((p / fps, "snare", fs_[p] / (fs_.max() + 1e-9)))
    sn = np.array(snares)
    hat_p = pick(fh, fps, 1.4, 0.05)
    hv = np.array([fh[p] for p in hat_p]) if hat_p else np.array([1.0])
    crash_thr = np.percentile(hv, 97) if len(hat_p) > 20 else np.inf
    for p in hat_p:
        if len(sn) and np.abs(sn - p).min() < int(0.035 * fps) and fh[p] < np.percentile(hv, 85):
            continue  # le souffle de la caisse claire n'est pas un charleston
        tail = hi_env[p + int(0.25 * fps):p + int(0.5 * fps)]
        crash = fh[p] >= crash_thr and len(tail) and tail.mean() > hi_env[p] * 0.25
        ev.append((p / fps, "crash" if crash else "hat", fh[p] / (hv.max() + 1e-9)))
    return sorted(ev)


# ---------- mélodique ----------
def transcribe_pitched(path, fmin, fmax, min_ms, onset=0.5, frame=0.3):
    from basic_pitch.inference import predict, ICASSP_2022_MODEL_PATH
    _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH, onset_threshold=onset, frame_threshold=frame,
                           minimum_note_length=min_ms, minimum_frequency=fmin, maximum_frequency=fmax,
                           multiple_pitch_bends=False, melodia_trick=True)
    return sorted((float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events)


def make_mono(notes):
    out = []
    for s, e, p, a in sorted(notes):
        if out and s < out[-1][1]:
            ps, pe, pp, pa = out[-1]
            if s - ps < 0.06 and a <= pa:  # note quasi simultanée plus faible : ignorée
                continue
            out[-1] = (ps, s, pp, pa)
            if s - ps < 0.06: out.pop()
        out.append((s, e, p, a))
    return out


def choose_program_other(rapport, override):
    if override is not None:
        return override, "imposé"
    rows = {r["label"]: r for r in rapport["par_source"].get("other", [])}
    best, fam = 0, None
    for name, (prog, labs) in FAMILIES.items():
        sc = max(((rows[l]["confiance_moy"] * (0.5 + rows[l]["pct_temps"] / 100)) for l in labs if l in rows), default=0)
        if sc > best: best, fam = sc, name
    if fam is None:
        return 89, "pad (aucun instrument net détecté sur 'other')"
    if fam == "guitare":
        el = rows.get("Electric guitar", {"confiance_moy": 0})["confiance_moy"]
        ac = rows.get("Acoustic guitar", {"confiance_moy": 0})["confiance_moy"]
        return (27, "guitare électrique") if el >= ac else (25, "guitare acoustique")
    return FAMILIES[fam][0], f"{fam} (score {best:.2f})"


# ---------- écriture MIDI ----------
class Grid:
    """Carte de tempo variable : un tempo par temps, déduit des temps détectés (lissés)."""

    def __init__(self, tempo):
        self.meter, self.period = tempo["meter"], tempo["period"]
        self.unit = 1.5 if self.meter == "12/8" else 1.0
        self.beat_ticks = PPQ * self.unit            # un temps = noire pointée en 12/8
        self.quant = 240 if self.meter == "12/8" else 120  # croche en 12/8, double croche en 4/4
        self.dur = tempo["dur"]
        b = tempo["beats"]
        self.PRE = int(np.ceil(max(b[0], 0) / self.period)) + 8   # la grille doit couvrir tout le morceau
        n_post = int(np.ceil(max(tempo["dur"] - b[-1], 0) / self.period)) + 8
        pre = b[0] - self.period * np.arange(self.PRE, 0, -1)
        post = b[-1] + self.period * np.arange(1, n_post + 1)
        self.B = np.concatenate([pre, b, post])
        idx = np.arange(len(self.B))
        ok = ((idx - self.PRE - tempo["downbeat"]) % 4 == 0) & (self.B <= 0)  # (downbeat compté sur les temps détectés)
        self.j_ref = int(idx[ok].max()) if ok.any() else 0   # dernier temps 1 avant le début de l'audio
        self.lead = float(-self.B[self.j_ref])               # secondes de MIDI avant le début de l'audio
        self.us_per_q = [int(round(d * 1e6 / self.unit)) for d in np.diff(self.B[self.j_ref:])]

    def u(self, t):
        return np.interp(t, self.B, np.arange(len(self.B))) - self.j_ref

    def raw(self, t):    # timing fidèle
        return int(round(float(self.u(t)) * self.beat_ticks))

    def q(self, t):      # calé sur la grille
        return int(round(float(self.u(t)) * self.beat_ticks / self.quant) * self.quant)


def meta_track(grid, name):
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    tr.append(mido.MetaMessage("set_tempo", tempo=grid.us_per_q[0], time=0))
    if grid.meter == "12/8":
        tr.append(mido.MetaMessage("time_signature", numerator=12, denominator=8, clocks_per_click=36, notated_32nd_notes_per_beat=8, time=0))
    else:
        tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    for us in grid.us_per_q[1:]:  # un changement de tempo par temps
        tr.append(mido.MetaMessage("set_tempo", tempo=us, time=int(grid.beat_ticks)))
    return tr


def note_track(name, channel, program, notes_ticks):
    """notes_ticks : liste (start_tick, end_tick, pitch, velocity)."""
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        tr.append(mido.Message("program_change", channel=channel, program=program, time=0))
    msgs = []
    for s, e, p, v in notes_ticks:
        msgs.append((s, 1, mido.Message("note_on", channel=channel, note=p, velocity=v)))
        msgs.append((e, 0, mido.Message("note_off", channel=channel, note=p, velocity=0)))
    msgs.sort(key=lambda m: (m[0], m[1]))
    last = 0
    for t, _, m in msgs:
        tr.append(m.copy(time=t - last)); last = t
    return tr


def save(path, grid, tracks, title):
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    mid.tracks.append(meta_track(grid, title))
    mid.tracks.extend(tracks)
    mid.save(path)


def vel(a):
    return int(np.clip(30 + a * 100, 1, 127))


def build(grid, tracks_data, quantized):
    """tracks_data: liste de (nom, canal, programme, [(start_s,end_s,pitch,vel)], mono)."""
    out = []
    f = grid.q if quantized else grid.raw
    for name, ch, prog, notes, mono in tracks_data:
        tn = []
        for s, e, p, v in notes:
            ts, te = f(s), f(e)
            if quantized:
                te = max(te, ts + grid.quant)
            else:
                te = max(te, ts + 10)
            if ts < 0: continue
            tn.append([ts, te, p, v])
        tn.sort()
        if quantized and name == "drums":  # doublons après quantification (même note, même tick)
            seen, uniq = set(), []
            for x in tn:
                if (x[0], x[2]) not in seen: seen.add((x[0], x[2])); uniq.append(x)
            tn = uniq
        if mono:
            for i in range(len(tn) - 1):
                if tn[i][1] > tn[i + 1][0]: tn[i][1] = tn[i + 1][0]
            tn = [x for x in tn if x[1] > x[0]]
        out.append((name, ch, prog, tn))
    return out


def process(name, out_root, args):
    d = out_root / name
    stems = d / "stems"
    rap = d / "rapport.json"
    if not rap.exists() or not all((stems / f"{s}.wav").exists() for s in ("vocals", "drums", "bass", "other")):
        print(f"  ! {name} : lancer d'abord ./analyse.sh (pistes ou rapport.json manquants)"); return None
    rapport = json.loads(rap.read_text(encoding="utf-8"))
    mdir = d / ("midi_extrait" if args.extrait else "midi")
    mdir.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="midi_"))
    wavs, mono = {}, {}
    for s in ("vocals", "drums", "bass", "other"):
        y, _ = librosa.load(stems / f"{s}.wav", sr=SR, mono=True,
                            offset=args.extrait[0] if args.extrait else 0.0, duration=args.extrait[1] if args.extrait else None)
        mono[s] = y
        p = tmp / f"{s}.wav"; sf.write(p, y, SR); wavs[s] = p
    mix = sum(mono.values())
    print("  tempo / mesure…")
    tempo = analyse_tempo(mono["drums"], mix)
    grid = Grid(tempo)
    print(f"    {tempo['meter']}, temps {60 / tempo['period']:.1f}/min, downbeat={tempo['downbeat']}, tempo {tempo['bpm_p5']}–{tempo['bpm_p95']}")
    print("  batterie…")
    dr = transcribe_drums(mono["drums"])
    drum_notes = [(t, t + 0.1, DRUM[k], vel(a)) for t, k, a in dr]
    print("  voix…")
    voc = make_mono(transcribe_pitched(wavs["vocals"], 80, 1100, 110))
    print("  basse…")
    bas = make_mono(transcribe_pitched(wavs["bass"], 30, 400, 90, onset=0.5, frame=0.3))
    print("  autres (polyphonique)…")
    oth = transcribe_pitched(wavs["other"], 70, 2000, 120, onset=0.55, frame=0.35)
    prog_other, why = choose_program_other(rapport, args.other_program)
    data = [
        ("vocals", 0, 73, [(s, e, p, vel(a)) for s, e, p, a in voc], True),
        ("bass", 1, 33, [(s, e, p, vel(a)) for s, e, p, a in bas], True),
        ("other", 2, prog_other, [(s, e, p, vel(a)) for s, e, p, a in oth], False),
        ("drums", 9, None, drum_notes, False),
    ]
    counts = {n: len(nt) for n, _, _, nt, _ in data}
    drum_counts = {k: sum(1 for e in dr if e[1] == k) for k in DRUM}
    for old in list(mdir.glob("*.mid")) + list(mdir.glob("*.wav")) + list(mdir.glob("partition.*")):  # un seul .mid par titre : on repart d'un dossier propre
        old.unlink()
    quant = args.version == "quantifie"
    built = build(grid, data, quant)
    midi_path = mdir / f"{name}.mid"
    save(midi_path, grid, [note_track(n, c, p, t) for n, c, p, t in built], name)
    if args.controle:
        wav = mdir / "controle.wav"
        subprocess.run(["fluidsynth", "-ni", "-q", "-r", "44100", "-F", str(wav), str(SOUNDFONT), str(midi_path)],
                       check=True, capture_output=True)
        y, sr_ = sf.read(wav)
        sf.write(wav, y[int(round(grid.lead * sr_)):int(round((grid.lead + grid.dur) * sr_))], sr_)  # le rendu démarre comme l'original
    res = {"morceau": name, "extrait": args.extrait, "tempo": {**{k: v for k, v in tempo.items() if k != "beats"}, "bpm_temps": round(60 / tempo["period"], 1),
           "bpm_noire": round(60 / tempo["period"] * (1.5 if tempo["meter"] == "12/8" else 1), 1)},
           "lead": round(grid.lead, 4), "version": args.version, "notes": counts, "batterie": drum_counts, "programme_other": prog_other, "choix_other": why}
    (mdir / "rapport_midi.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    t = res["tempo"]
    (mdir / "rapport_midi.md").write_text("\n".join([
        f"# MIDI : {name}", "", f"- Mesure détectée : **{t['meter']}**, tempo **{t['bpm_temps']} temps/min** "
        f"({'noire pointée' if t['meter'] == '12/8' else 'noire'}) = {t['bpm_noire']} noires/min dans le fichier MIDI",
        f"- Variation du tempo (carte de tempo, un tempo par temps) : de {t['bpm_p5']} à {t['bpm_p95']} temps/min (5e–95e percentile), {t['n_beats']} temps détectés",
        f"- Score mesure : 4/4 = {t['scores']['4/4']}, 12/8 = {t['scores']['12/8']}",
        f"- Programme piste « other » : {prog_other} — {why}", "", "## Nombre de notes", "",
        *[f"- {k} : {v}" for k, v in counts.items()], "", "Batterie : " + ", ".join(f"{k} {v}" for k, v in drum_counts.items()), "",
        f"Fichier : `{name}.mid` (version {'quantifiée' if quant else 'brute'}, 4 pistes : vocals, bass, other, drums)" + (" ; `controle.wav` = rendu audio pour comparer à l'original." if args.controle else ".")]) + "\n",
        encoding="utf-8")
    return res


def compute_lead(name, out_root):
    """Secondes de MIDI avant le début de l'audio, pour les MIDI générés avant que 'lead' soit sauvegardé (même calcul que process)."""
    stems = Path(out_root) / name / "stems"
    mono = {s: librosa.load(stems / f"{s}.wav", sr=SR, mono=True)[0] for s in ("vocals", "drums", "bass", "other")}
    return round(Grid(analyse_tempo(mono["drums"], sum(mono.values()))).lead, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--morceau", help="nom d'un seul morceau (dossier de output/)")
    ap.add_argument("--extrait", help="DEBUT:DUREE en secondes, pour tester (écrit dans midi_extrait/)")
    ap.add_argument("--other-program", type=int, help="programme GM imposé pour la piste 'other' (ex. 27)")
    ap.add_argument("--version", choices=["quantifie", "brut"], default="quantifie", help="timing du fichier MIDI unique")
    ap.add_argument("--sans-controle", action="store_true", help="ne pas générer controle.wav")
    a = ap.parse_args()
    a.controle = not a.sans_controle
    a.extrait = tuple(float(x) for x in a.extrait.split(":")) if a.extrait else None
    out = Path(a.output)
    if not SOUNDFONT.exists(): sys.exit(f"Soundfont absente : {SOUNDFONT}")
    names = [a.morceau] if a.morceau else sorted(p.name for p in out.iterdir() if p.is_dir())
    results = []
    for n in names:
        print(f"\n=== {n} ===")
        r = process(n, out, a)
        if r: results.append(r)
    if results and not a.extrait:
        L = ["# Synthèse MIDI", "", "| Morceau | Mesure | Temps/min | Voix | Basse | Other | Batterie | Programme other |", "|---|---|---|---|---|---|---|---|"]
        for r in results:
            t, c = r["tempo"], r["notes"]
            L.append(f"| {r['morceau']} | {t['meter']} | {t['bpm_temps']} | {c['vocals']} | {c['bass']} | {c['other']} | {c['drums']} | {r['programme_other']} ({r['choix_other']}) |")
        (out / "midi_synthese.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
