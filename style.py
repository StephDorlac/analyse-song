"""Export style Yamaha (.sty, SFF GE) pour DGX-670, à partir du MIDI généré : batterie, basse, guitare/harmonie (Intro A/B, Main A/B, Fill In AA/BB, Ending A/B)."""
import argparse, io, json, struct, unicodedata
from bisect import bisect_right
from pathlib import Path
import mido
from score import read_midi, detect_chords

ROOT = Path(__file__).resolve().parent
PPQ_IN, PPQ_OUT = 480, 1920          # le MIDI du projet est à 480, les styles Yamaha à 1920
SCALE = PPQ_OUT // PPQ_IN
DRUM_CH, BASS_CH, CHORD_CH = 9, 10, 12   # canaux Yamaha (0-based) : Rhythm 1 (batterie), Bass, Chord 2 (guitare / harmonie)
BASS_ROOT, CHORD_ROOT = 24, 48            # fondamentale de l'accord source (C) dans le registre de la basse / de l'harmonie
# degré de l'accord source Cmaj7 pour chaque intervalle au-dessus de la fondamentale : le clavier remet tierce, septième… selon l'accord joué
DEGREE = {0: 0, 1: 0, 2: 2, 3: 4, 4: 4, 5: 5, 6: 7, 7: 7, 8: 9, 9: 9, 10: 11, 11: 11}

# entrées Ctb2 (47 octets) reprises telles quelles d'un style SFF GE qui fonctionne sur DGX (accord source Cmaj7, règles de transposition basse/batterie)
CTB2 = {DRUM_CH: bytes.fromhex("094d61696e4472756d09000fff07ffffffff0002007f000000007f00010006007f01000000007f0000010018515a00"),
        BASS_CH: bytes.fromhex("0a42617373202020200a010fff03ffffffff0002007f000000007f000081041c7f02000000007f0000000000800000"),
        CHORD_CH: b"\x0cE.Guitar\x0c\x01\x0f\xff\x03\xff\xff\xff\xff\x00\x02\x00\x7f\x00\x00\x00\x00\x7f\x00\x02\x81\x02\x00\x7f\x01\x00\x00\x00\x00\x7f\x00\x00\x00\x00\x00\x80\x00\x00"}
assert all(len(b) == 47 for b in CTB2.values())
# SysEx Yamaha de l'en-tête et réglages d'effets, repris du même style
SYSEX_HEAD = ["43761a100101010101010101", "43733911004600", "43730151050001080000000000000000", "43730151050002080000000000000000"]
SYSEX_FX = ["43104c0201000113", "43104c0201204210", "43104c0201401612", "43104c0201460047", "43104c02015a01", "43104c08080703", "43104c08090702"]
# (CC0 banque MSB, CC32 LSB, programme, volume, reverb) par canal ; le programme de l'harmonie vient du rapport MIDI
VOICES = {DRUM_CH: (127, 0, 0, 100, 5), BASS_CH: (0, 0, 33, 100, 0), CHORD_CH: (0, 0, None, 90, 20)}


# ---------- lecture ----------
def bar_len(sig):
    return int(sig[0] * 4 / sig[1] * PPQ_IN)


def load(stem_dir, stem):
    tracks, sig = read_midi(stem_dir / f"{stem}.mid")
    for need in ("drums", "bass"):
        if need not in tracks: raise ValueError(f"pas de piste « {need} » dans le MIDI")
    return tracks, sig


# ---------- choix des sections ----------
def signature(tracks, s, n, bl):
    """Empreinte rythmique d'une fenêtre de n mesures commençant à la mesure s (position dans la fenêtre, élément)."""
    lo, hi = s * bl, (s + n) * bl
    sig = {("d", t - lo, p) for t, _, p, _ in tracks["drums"] if lo <= t < hi}
    return sig | {("b", t - lo) for t, _, _, _ in tracks["bass"] if lo <= t < hi}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def count(notes, lo, hi):
    return sum(1 for t, *_ in notes if lo <= t < hi)


def pick_windows(tracks, nb, n, bl):
    """Main A = fenêtre la plus représentative du morceau (la plus proche des autres) ; Main B = fenêtre plus dense, distincte."""
    starts = [s for s in range(1, nb - n)]
    dr = {s: count(tracks["drums"], s * bl, (s + n) * bl) for s in starts}
    ba = {s: count(tracks["bass"], s * bl, (s + n) * bl) for s in starts}
    med_d = sorted(dr.values())[len(dr) // 2] if dr else 0
    med_b = sorted(ba.values())[len(ba) // 2] if ba else 0
    ok = [s for s in starts if dr[s] >= 0.5 * med_d and ba[s] >= 0.3 * med_b and dr[s] > 0]
    if not ok: raise ValueError("aucune fenêtre exploitable (batterie ou basse trop pauvres)")
    sigs = {s: signature(tracks, s, n, bl) for s in starts}

    def typical(s, among):
        o = [jaccard(sigs[s], sigs[t]) for t in among if abs(t - s) >= n]
        return sum(o) / len(o) if o else 0.0

    a = max(ok, key=lambda s: typical(s, ok))
    dense = [s for s in ok if abs(s - a) >= n and dr[s] >= 1.1 * dr[a]]
    dense = sorted(dense, key=lambda s: -dr[s])[:max(1, len(dense) // 2)]
    b = max(dense, key=lambda s: typical(s, ok)) if dense else None
    return a, b


def pick_fills(tracks, nb, bl, taken):
    """Mesures de roulement : beaucoup de caisse claire / cymbale, de préférence juste avant un crash sur le premier temps."""
    cand = []
    for m in range(1, nb - 1):
        if any(lo <= m < hi for lo, hi in taken): continue
        lo, hi = m * bl, (m + 1) * bl
        hits = sum(1 for t, _, p, _ in tracks["drums"] if lo <= t < hi and p in (38, 49))
        crash_next = any(p == 49 and hi <= t < hi + 120 for t, _, p, _ in tracks["drums"])
        cand.append((hits + (4 if crash_next else 0), m))
    cand.sort(key=lambda x: (-x[0], x[1]))
    return [m for _, m in cand[:2]]


def pick_intro_ending(tracks, nb, bl, n_a=2, n_b=4):
    """Intro : les mesures juste avant l'entrée de la batterie (si elles jouent quelque chose), sinon celles où elle entre.
    Ending : les dernières mesures où la batterie joue. Retourne {nom: (mesure de départ, mesures)}."""
    per = {}
    for t, *_ in tracks["drums"]: per[t // bl] = per.get(t // bl, 0) + 1
    fb = next((m for m in sorted(per) if per[m] >= 3 and per.get(m + 1, 0) >= 3), 0)
    last = max(per)
    plan = {}
    for tag, n in (("A", n_a), ("B", n_b)):
        n = min(n, nb - 1)
        pre = max(fb - n, 0)
        busy = count(tracks["bass"], pre * bl, fb * bl) + count(tracks.get("other", []), pre * bl, fb * bl)
        plan[f"Intro {tag}"] = (pre, n) if fb - pre == n and busy >= 4 else (fb, min(n, nb - fb))
        plan[f"Ending {tag}"] = (max(last + 1 - n, 0), n)
    return plan


# ---------- conversion ----------
def bass_reference(tracks):
    """Hauteur plancher du registre de basse du morceau (10e centile) : la fondamentale se place au-dessus."""
    ps = sorted(p for _, _, p, _ in tracks["bass"])
    return ps[len(ps) // 10]


def to_source(p, root_pc, floor):
    """Note de basse -> note équivalente sur l'accord source (C) : intervalle depuis la fondamentale, registre conservé."""
    r = floor + (root_pc - floor) % 12   # fondamentale la plus basse au-dessus du plancher
    d = p - r
    return BASS_ROOT + 12 * (d // 12) + DEGREE[d % 12]


def chord_to_source(p, root_pc, floor):
    """Note d'harmonie -> équivalent sur l'accord source (C) : même principe que la basse, dans le registre de l'harmonie."""
    r = floor + (root_pc - floor) % 12
    d = p - r
    return CHORD_ROOT + 12 * (d // 12) + DEGREE[d % 12]


def section_notes(tracks, chords, floor, start_bar, n_bars, bl, offset, chord_floor=None):
    """Notes batterie + basse + harmonie de la fenêtre, décalées à `offset` (ticks sortie), ramenées sur l'accord source."""
    lo, hi = start_bar * bl, (start_bar + n_bars) * bl
    times = [c[0] for c in chords]
    out = []
    for t, e, p, v in tracks["drums"]:
        if lo <= t < hi:
            out.append((offset + (t - lo) * SCALE, offset + (min(e, hi) - lo) * SCALE, DRUM_CH, p, v))
    for t, e, p, v in tracks["bass"]:
        if lo <= t < hi:
            root = chords[max(bisect_right(times, t) - 1, 0)][1]
            out.append((offset + (t - lo) * SCALE, offset + (min(e, hi) - lo) * SCALE, BASS_CH, to_source(p, root, floor), v))
    if chord_floor is not None:
        seen = set()
        for t, e, p, v in tracks["other"]:
            if lo <= t < hi:
                root = chords[max(bisect_right(times, t) - 1, 0)][1]
                q = chord_to_source(p, root, chord_floor)
                if (t, q) in seen: continue   # deux notes de l'accord réel peuvent tomber sur le même degré
                seen.add((t, q))
                out.append((offset + (t - lo) * SCALE, offset + (min(e, hi) - lo) * SCALE, CHORD_CH, q, v))
    return out


# ---------- écriture ----------
def ascii32(name):
    return (unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()[:27] + ".STY").ljust(32, "\0")


def chunk(cid, body):
    return cid + struct.pack(">I", len(body)) + body


def casm(sections, channels):
    """Un CSEG par section : la même règle de transposition par canal."""
    segs = b""
    for name in sections:
        body = chunk(b"Sdec", name.encode("ascii"))
        for ch in channels:
            body += chunk(b"Ctb2", CTB2[ch])
        segs += chunk(b"CSEG", body)
    return chunk(b"CASM", segs)


def header_events(sig, bpm, title, chord_prog):
    ev = [mido.MetaMessage("time_signature", numerator=sig[0], denominator=sig[1], clocks_per_click=36 if sig == (12, 8) else 24, notated_32nd_notes_per_beat=8),
          mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)),
          mido.MetaMessage("marker", text="SFF2"),
          mido.MetaMessage("track_name", name=ascii32(title))]
    ev += [mido.Message("sysex", data=bytes.fromhex(h)) for h in SYSEX_HEAD]
    ev += [mido.MetaMessage("marker", text="SInt"), mido.Message("sysex", data=bytes.fromhex("7e7f0901"))]
    for ch, (msb, lsb, prog, vol, rev) in VOICES.items():
        if prog is None and chord_prog is None: continue
        prog = chord_prog if prog is None else prog
        ev += [mido.Message("control_change", channel=ch, control=7, value=vol), mido.Message("control_change", channel=ch, control=91, value=rev),
               mido.Message("control_change", channel=ch, control=0, value=msb), mido.Message("control_change", channel=ch, control=32, value=lsb),
               mido.Message("program_change", channel=ch, program=prog)]
    return ev + [mido.Message("sysex", data=bytes.fromhex(h)) for h in SYSEX_FX]


def write_style(path, sig, bpm, title, layout, chord_prog=None):
    """layout : liste (nom de section, notes [(début, fin, canal, note, vélocité)], longueur en ticks)."""
    bl = bar_len(sig) * SCALE
    timed = [(0, 0, m) for m in header_events(sig, bpm, title, chord_prog)]
    t = bl                                  # une mesure d'initialisation (SInt) avant la première section
    for name, notes, length in layout:
        timed += [(t, 0, mido.MetaMessage("marker", text=name)), (t, 1, mido.MetaMessage("text", text=f"fn:{name}\0"))]
        for s, e, ch, p, v in notes:
            e = max(e, s + 10)
            timed += [(s + t, 3, mido.Message("note_on", channel=ch, note=p, velocity=v)), (e + t, 2, mido.Message("note_off", channel=ch, note=p, velocity=0))]
        t += length
    timed.append((t, 4, mido.MetaMessage("end_of_track")))
    timed.sort(key=lambda x: (x[0], x[1]))
    tr, last = mido.MidiTrack(), 0
    for at, _, m in timed:
        tr.append(m.copy(time=at - last)); last = at
    mid = mido.MidiFile(type=0, ticks_per_beat=PPQ_OUT)
    mid.tracks.append(tr)
    buf = io.BytesIO(); mid.save(file=buf)
    Path(path).write_bytes(buf.getvalue() + casm([n for n, _, _ in layout], [DRUM_CH, BASS_CH] + ([CHORD_CH] if chord_prog is not None else [])))


def build_style(stem_dir, stem, out_path, bars=4):
    """Écrit le .sty et retourne un rapport (sections retenues, tempo, nombre de notes)."""
    stem_dir = Path(stem_dir)
    tracks, sig = load(stem_dir, stem)
    info = json.loads((stem_dir / "rapport_midi.json").read_text(encoding="utf-8"))
    bpm = int(round(info["tempo"]["bpm_noire"]))
    if not 32 <= bpm <= 280: raise ValueError(f"tempo {bpm} hors des limites du clavier (32–280)")
    bl = bar_len(sig)
    nb = max(e for n in ("drums", "bass") for _, e, _, _ in tracks[n]) // bl
    if nb < 2 * bars + 4: raise ValueError("morceau trop court pour en extraire un style")
    a, b = pick_windows(tracks, nb, bars, bl)
    taken = [(a, a + bars)] + ([(b, b + bars)] if b is not None else [])
    fills = pick_fills(tracks, nb, bl, taken)
    chords = detect_chords(tracks, sig, False)
    if not chords: raise ValueError("aucun accord détecté")
    floor = bass_reference(tracks)
    other = tracks.get("other", [])
    chord_floor = sorted(p for _, _, p, _ in other)[len(other) // 10] if other else None
    chord_prog = info.get("programme_other") if other else None
    if chord_prog is None: chord_floor = None
    main_b = b if b is not None else a
    ie = pick_intro_ending(tracks, nb, bl)
    plan = [("Intro A", *ie["Intro A"]), ("Intro B", *ie["Intro B"]), ("Main A", a, bars), ("Main B", main_b, bars),
            ("Fill In AA", fills[0], 1), ("Fill In BB", fills[-1], 1), ("Ending A", *ie["Ending A"]), ("Ending B", *ie["Ending B"])]
    layout = [(name, section_notes(tracks, chords, floor, s, n, bl, 0, chord_floor), n * bl * SCALE) for name, s, n in plan]
    write_style(out_path, sig, bpm, stem, layout, chord_prog)
    return {"morceau": stem, "mesure": f"{sig[0]}/{sig[1]}", "tempo": bpm, "mesures_par_section": bars,
            "sections": [{"nom": nm, "depart_mesure": s + 1, "mesures": n, "notes": len(nt)} for (nm, s, n), (_, nt, _) in zip(plan, layout)],
            "main_b_distinct": b is not None, "harmonie": None if chord_floor is None else {"programme_gm": chord_prog, "canal": CHORD_CH + 1}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--morceau", required=True)
    ap.add_argument("--mesures", type=int, default=4, help="longueur des Main A/B en mesures")
    a = ap.parse_args()
    d = Path(a.output) / a.morceau / "midi"
    print(json.dumps(build_style(d, a.morceau, d / f"{a.morceau}.sty", a.mesures), ensure_ascii=False, indent=2))
