"""MIDI (output/<morceau>/midi/<morceau>.mid) -> partition MusicXML affichable dans l'interface."""
import json, re, sys, time
from pathlib import Path
import mido
from music21 import stream, note, chord, meter, key, tempo, clef, duration, instrument, layout, harmony, pitch as m21pitch

PPQ = 480
SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
# batterie : note GM -> (hauteur affichée sur la portée percussion, tête de note)
DRUM_DISPLAY = {36: ("F4", "normal"), 38: ("C5", "normal"), 42: ("G5", "x"), 49: ("A5", "x")}
NAMES = {"vocals": "Voix", "bass": "Basse", "other": "Autres (harmonie)", "drums": "Batterie"}


def read_midi(path):
    """Retourne (tracks {nom: [(start, end, pitch, vel)]}, signature (num, den))."""
    mid = mido.MidiFile(path)
    sig, tracks = (4, 4), {}
    for tr in mid.tracks:
        t, open_, notes = 0, {}, []
        for m in tr:
            t += m.time
            if m.type == "time_signature":
                sig = (m.numerator, m.denominator)
            elif m.type == "note_on" and m.velocity > 0:
                open_[(m.channel, m.note)] = (t, m.velocity)
            elif m.type in ("note_off", "note_on") and (m.channel, m.note) in open_:
                s, v = open_.pop((m.channel, m.note))
                notes.append((s, max(t, s + 1), m.note, v))
        if notes:
            tracks[tr.name] = sorted(notes)
    return tracks, sig


def tick_to_sec(path):
    """Fonction ticks -> secondes (temps MIDI, avant le décalage 'lead') d'après la carte de tempo du fichier."""
    mid = mido.MidiFile(path)
    ev, t = [], 0
    for m in mid.tracks[0]:
        t += m.time
        if m.type == "set_tempo":
            ev.append((t, m.tempo))
    pts, sec = [], 0.0
    for i, (tk, us) in enumerate(ev):
        pts.append((tk, sec, us))
        if i + 1 < len(ev):
            sec += (ev[i + 1][0] - tk) / mid.ticks_per_beat * us / 1e6

    def f(tick):
        base = pts[0]
        for p in pts:
            if p[0] <= tick: base = p
            else: break
        return base[1] + (tick - base[0]) / mid.ticks_per_beat * base[2] / 1e6
    return f


def assign_words(times, words):
    """Associe chaque mot à la note dont l'attaque en est la plus proche (une note = au plus un mot, ordre conservé).
    times : instants des notes (s, audio). Retourne {index de note: mot}."""
    n, m, SKIP = len(words), len(times), 0.6
    INF = float("inf")
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]   # alignement monotone mots/notes (programmation dynamique)
    for i in range(1, n + 1):
        cost[i][0] = i * SKIP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dt = abs(times[j - 1] - words[i - 1]["debut"])
            cost[i][j] = min(cost[i - 1][j] + SKIP, cost[i][j - 1], cost[i - 1][j - 1] + dt if dt <= 0.5 else INF)
    where, i, j = {}, n, m   # index de mot -> index de note
    while i > 0 and j > 0:
        dt = abs(times[j - 1] - words[i - 1]["debut"])
        if dt <= 0.5 and cost[i][j] == cost[i - 1][j - 1] + dt:
            where[i - 1] = j - 1; i -= 1; j -= 1
        elif cost[i][j] == cost[i][j - 1]: j -= 1
        else: i -= 1
    # mots restés sans note (souvent le début : voix non détectée) : note libre la plus proche, entre les notes des mots voisins, à 2 s max
    FAR, used, prev = 2.0, set(where.values()), -1
    for w in range(n):
        if w in where:
            prev = where[w]; continue
        nxt = next((where[k] for k in range(w + 1, n) if k in where), m)
        cand = [j for j in range(prev + 1, nxt) if j not in used and abs(times[j] - words[w]["debut"]) <= FAR]
        if cand:
            where[w] = prev = min(cand, key=lambda j: abs(times[j] - words[w]["debut"])); used.add(prev)
    # encore sans note (voix non transcrite) : le mot est accolé à la note du mot voisin le plus proche dans le temps, à 3 s max
    glued = {}
    for w in range(n - 1, -1, -1):   # de droite à gauche : une levée (« I found ») se colle à la note du mot suivant
        if w in where:
            continue
        for k in (w + 1, w - 1):
            note = where.get(k, glued.get(k))
            if 0 <= k < n and note is not None and abs(times[note] - words[w]["debut"]) <= 3.0:
                glued[w] = note; break
    lyr = {}
    for w, j in sorted({**where, **glued}.items()):
        lyr[j] = (lyr[j] + " " if j in lyr else "") + words[w]["m"]
    return lyr


QUALITIES = {  # nom music21 -> (intervalles, suffixe US, pénalité de complexité)
    "major": ((0, 4, 7), "", 0.0), "minor": ((0, 3, 7), "m", 0.0), "dominant-seventh": ((0, 4, 7, 10), "7", 0.07),
    "major-seventh": ((0, 4, 7, 11), "maj7", 0.07), "minor-seventh": ((0, 3, 7, 10), "m7", 0.07), "diminished": ((0, 3, 6), "dim", 0.05),
}


def detect_chords(tracks, sig, flats):
    """Accords par temps d'après les pitch classes de « autres » + basse. Retourne [(tick, racine, qualité)] aux changements."""
    import numpy as np
    beat = 720 if sig == (12, 8) else 480
    end = max((e for n in ("other", "bass") for _, e, _, _ in tracks.get(n, [])), default=0)
    nb = end // beat + 1
    V, B = np.zeros((nb, 12)), np.zeros((nb, 12))
    for name, M, wgt in (("other", V, 1.0), ("bass", V, 2.0)):
        for st, en, p, v in tracks.get(name, []):
            for b in range(st // beat, min((en - 1) // beat, nb - 1) + 1):
                ov = min(en, (b + 1) * beat) - max(st, b * beat)
                if ov > 0:
                    M[b, p % 12] += wgt * ov * v / 127
                    if name == "bass": B[b, p % 12] += ov
    Vs = V.copy()  # lissage : le temps courant compte double par rapport aux voisins
    Vs[1:] += 0.5 * V[:-1]; Vs[:-1] += 0.5 * V[1:]
    seq, prev = [], None
    for b in range(nb):
        v, tot = Vs[b], Vs[b].sum()
        if tot <= 0: seq.append(prev); continue
        best, bs = None, -9
        for q, (iv, _, pen) in QUALITIES.items():
            for r in range(12):
                tones = {(r + i) % 12 for i in iv}
                inn = sum(v[t] for t in tones); out = tot - inn
                sc = inn / tot - 0.5 * out / tot - pen + (0.25 if B[b].any() and int(B[b].argmax()) == r else 0) + (0.08 if prev == (r, q) else 0)
                if len(iv) == 4 and v[(r + iv[3]) % 12] < 0.12 * tot: sc -= 0.1  # septième absente : accord simple
                if sc > bs: bs, best = sc, (r, q)
        prev = best if bs >= 0.35 else prev
        seq.append(prev)
    for b in range(1, nb - 1):  # un accord d'un seul temps entouré du même accord est du bruit
        if seq[b] != seq[b - 1] and seq[b - 1] == seq[b + 1]: seq[b] = seq[b - 1]
    out, last = [], None
    for b, c in enumerate(seq):
        if c is not None and c != last: out.append((b * beat, *c))
        last = c if c is not None else last
    return out


def quantize(notes, g):
    out = []
    for s, e, p, v in notes:
        qs = int(round(s / g) * g)
        out.append((qs, max(int(round(e / g) * g), qs + g), p, v))
    return out


def detect_key(tracks):
    """Tonalité estimée sur basse + autres (algorithme de Krumhansl) ; retourne music21 Key."""
    s = stream.Stream()
    for name in ("other", "bass"):
        for st, en, p, v in tracks.get(name, [])[:1500]:
            n = note.Note(p); n.quarterLength = max((en - st) / PPQ, 0.25); s.insert(st / PPQ, n)
    return s.analyze("key")


def pname(p, flats):
    return (FLAT if flats else SHARP)[p % 12] + str(p // 12 - 1)


def build_part(name, notes, g, flats, ts, clef_obj, drum=False, mono=False, cap=None, words=None, secs=None):
    """Regroupe les notes qui débutent ensemble en accord ; durée limitée à l'attaque suivante."""
    part = stream.Part()
    part.partName = NAMES[name]
    part.insert(0, clef_obj)
    part.insert(0, meter.TimeSignature(ts))
    by_start = {}
    for s, e, p, v in notes:
        by_start.setdefault(s, []).append((e, p))
    starts = sorted(by_start)
    lyr = assign_words([secs[s] for s in starts], words) if words and secs else {}
    for i, s in enumerate(starts):
        ends = [e for e, _ in by_start[s]]
        d = (min(ends) if mono or drum else max(ends)) - s
        if i + 1 < len(starts):
            d = min(d, starts[i + 1] - s)
        if cap:
            d = min(d, cap)
        d = max(d, g)
        if drum:
            hits = sorted({DRUM_DISPLAY[p] for _, p in by_start[s] if p in DRUM_DISPLAY})
            if not hits: continue
            ns = []
            for pn, head in hits:
                n = note.Note(pn); n.notehead = head; ns.append(n)
            el = ns[0] if len(ns) == 1 else chord.Chord(ns)
        else:
            ps = sorted({p for _, p in by_start[s]})
            if mono: ps = ps[-1:]
            el = note.Note(pname(ps[0], flats)) if len(ps) == 1 else chord.Chord([pname(p, flats) for p in ps])
        el.duration = duration.Duration(d / PPQ)
        if i in lyr: el.lyric = lyr[i]
        part.insert(s / PPQ, el)
    return part


def chord_sequence(chords, sig, n_meas, flats, midi_path, info, segments):
    """Accords dans l'ordre du morceau, pour la transposition : mesure, temps, fondamentale, suffixe, intervalles, paroles chantées dessus, début/fin (s) dans l'audio."""
    beat = 720 if sig == (12, 8) else 480
    mlen = int(sig[0] * 4 / sig[1] * PPQ)
    words = [w for sg in (segments or []) for w in sg.get("mots", [])] if info and "lead" in info else []
    t2s = tick_to_sec(midi_path) if info and "lead" in info else None
    end = n_meas * mlen
    out = []
    for i, (tk, r, q) in enumerate(chords):
        iv, suf, _ = QUALITIES[q]
        nxt = chords[i + 1][0] if i + 1 < len(chords) else end
        lyr, a, b = "", None, None
        if t2s:
            a, b = t2s(tk) - info["lead"], t2s(nxt) - info["lead"]
        if words:
            lyr = " ".join(w["m"].strip() for w in words if a <= w["debut"] < b)
        out.append({"mesure": tk // mlen + 1, "temps": round((tk % mlen) / beat, 2), "racine": r, "duree": (nxt - tk) // beat, "suffixe": suf, "intervalles": list(iv), "paroles": lyr, "debut": None if a is None else round(a, 3), "fin": None if b is None else round(b, 3)})
    return out


def build_score(midi_path, out_path, info=None, segments=None):
    """segments : segments de paroles.json (avec 'mots' horodatés) ou None. info['lead'] : décalage MIDI/audio (s)."""
    t0 = time.time()
    tracks, sig = read_midi(midi_path)
    words, secs = [], {}
    if segments and "vocals" in tracks and info and "lead" in info:
        words = [w for sg in segments for w in sg.get("mots", [])]
        t2s = tick_to_sec(midi_path)
        raw_starts = [st for st, _, _, _ in tracks["vocals"]]
    ts = f"{sig[0]}/{sig[1]}"
    g = 240 if sig == (12, 8) else 120
    if words:  # instant audio de chaque attaque de voix, avant quantification (clé : attaque quantifiée)
        secs = {}
        for rs, (qs, _, _, _) in zip(raw_starts, quantize(tracks["vocals"], g)):
            secs.setdefault(qs, t2s(rs) - info["lead"])
    tracks = {k: quantize(v, g) for k, v in tracks.items()}
    k = detect_key(tracks)
    flats = k.sharps < 0
    beat = 720 if sig == (12, 8) else 480
    sc = stream.Score()
    sc.metadata = None
    parts = []
    if "vocals" in tracks: parts.append(build_part("vocals", tracks["vocals"], g, flats, ts, clef.TrebleClef(), mono=True, words=words, secs=secs))
    other = tracks.get("other", [])
    if other:
        up = build_part("other", [n for n in other if n[2] >= 60], g, flats, ts, clef.TrebleClef(), cap=beat * 2)
        lo = build_part("other", [n for n in other if n[2] < 60], g, flats, ts, clef.BassClef(), cap=beat * 2)
        up.partName = NAMES["other"]; lo.partName = ""
        parts += [up, lo]
        grp = layout.StaffGroup([up, lo], symbol="brace", name=NAMES["other"])
    if "bass" in tracks: parts.append(build_part("bass", tracks["bass"], g, flats, ts, clef.BassClef(), mono=True))
    if "drums" in tracks: parts.append(build_part("drums", tracks["drums"], g, flats, ts, clef.PercussionClef(), drum=True, cap=beat))
    chords = detect_chords(tracks, sig, flats)
    for tk, r, q in chords:  # symboles d'accords (notation US) au-dessus de la première portée
        parts[0].insert(tk / PPQ, harmony.ChordSymbol(root=(FLAT if flats else SHARP)[r].replace("b", "-"), kind=q))
    for p in parts:
        p.insert(0, key.KeySignature(k.sharps)) if p.partName != NAMES["drums"] else None
        sc.insert(0, p)
    if other: sc.insert(0, grp)
    if info:
        mm = tempo.MetronomeMark(number=info["tempo"]["bpm_temps"], referent=duration.Duration(1.5 if sig == (12, 8) else 1))
        parts[0].insert(0, mm)
    for p in parts:
        p.makeNotation(inPlace=True)
    sc.write("musicxml", fp=str(out_path))
    xml = Path(out_path).read_text(encoding="utf-8")  # music21 n'écrit pas le texte d'affichage des accords : on y met le suffixe US
    for q, (_, suf, _) in QUALITIES.items():
        xq = "dominant" if q == "dominant-seventh" else q  # nom écrit par music21
        if suf: xml = xml.replace(f"<kind>{xq}</kind>", f'<kind text="{suf}">{xq}</kind>')
    Path(out_path).write_text(xml, encoding="utf-8")
    n_meas = max(len(p.getElementsByClass("Measure")) for p in parts)
    groups = {}
    for i, p in enumerate(parts):  # regroupe les portées par instrument (les « autres » ont 2 portées : piano)
        groups.setdefault(p.partName or NAMES["other"], []).append(i)
    grille = None
    if info and "lead" in info:  # instant audio (s) de chaque noire de la partition : sert à synchroniser lecture et curseur
        t2s_ = tick_to_sec(midi_path)
        qpm = sig[0] * 4 / sig[1]
        grille = [round(t2s_(int(i * PPQ)) - info["lead"], 3) for i in range(int(n_meas * qpm) + 2)]
    seq = chord_sequence(chords, sig, n_meas, flats, midi_path, info, segments)
    resume = {}  # accords distincts : temps joués, pitch classes (fondamentale d'abord)
    for i, (tk, r, q) in enumerate(chords):
        nxt = chords[i + 1][0] if i + 1 < len(chords) else int(n_meas * sig[0] * 4 / sig[1] * PPQ)
        iv, suf, _ = QUALITIES[q]
        c = resume.setdefault((r, q), {"nom": ((FLAT if flats else SHARP)[r] + suf).replace("-", "b"), "notes": [(r + x) % 12 for x in iv], "fois": 0, "temps": 0})
        c["fois"] += 1; c["temps"] += (nxt - tk) // beat
    resume = sorted(resume.values(), key=lambda c: -c["temps"])
    return {"resume_accords": resume, "accords_seq": seq, "grille": grille, "parts": [{"nom": n, "indices": ix} for n, ix in groups.items()], "mesures": n_meas, "mesure": ts, "tonalite": f"{k.tonic.name.replace('-', 'b')} {k.mode}", "accords": len(chords), "mots": len(words), "mots_places": len(re.findall("<lyric", xml)), "secondes": round(time.time() - t0, 1)}


if __name__ == "__main__":
    d = Path(sys.argv[1]); name = d.parent.name
    info = json.loads((d / "rapport_midi.json").read_text(encoding="utf-8"))
    import paroles
    print(build_score(d / f"{name}.mid", d / "partition.musicxml", info, paroles.ensure_words(d.parent)))
