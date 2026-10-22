"""Pipeline : Demucs (séparation) + PANNs (tagging AudioSet) + rapports. Lancé par analyse.sh."""
import argparse, json, subprocess, sys, datetime
from pathlib import Path
import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent
SR = 32000          # fréquence attendue par PANNs Cnn14
WIN = 5 * SR        # fenêtres de 5 s
AUDIO_EXT = {".mp3", ".wav"}
STEMS = ["vocals", "drums", "bass", "other"]


def keep_index(i):
    # Speech (0-3), Singing (27-35), Musical instrument (139-215, sans le parent 138)
    return i <= 3 or 27 <= i <= 35 or 139 <= i <= 215


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    fmt = d.get("format", {})
    st = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    return {
        "duree_s": float(fmt.get("duration", 0)),
        "codec": st.get("codec_name"), "echantillonnage_hz": st.get("sample_rate"),
        "canaux": st.get("channels"), "debit_bps": fmt.get("bit_rate"),
        "taille_octets": int(fmt.get("size", 0)), "tags": tags,
    }


def separate(mp3, workdir, device):
    stem_dir = workdir / "stems"
    if all((stem_dir / f"{s}.wav").exists() for s in STEMS):
        return stem_dir
    tmp = workdir / "_demucs"
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", device, "-o", str(tmp), str(mp3)], check=True)
    src = tmp / "htdemucs" / mp3.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    for s in STEMS:
        (src / f"{s}.wav").replace(stem_dir / f"{s}.wav")
    return stem_dir


def tag(model, labels, audio):
    """Retourne {label: np.array(prob par fenêtre de 5 s)} pour les labels d'instruments/voix."""
    n = max(1, int(np.ceil(len(audio) / WIN)))
    chunks = []
    for k in range(n):
        c = audio[k * WIN:(k + 1) * WIN]
        if len(c) < SR:  # fenêtre finale trop courte : ignorée
            continue
        chunks.append(np.pad(c, (0, WIN - len(c))))
    probs = np.concatenate([model.inference(np.stack(chunks[i:i + 16]))[0] for i in range(0, len(chunks), 16)])
    return {labels[i]: probs[:, i] for i in range(len(labels)) if keep_index(i)}


def summarize(tags, thr):
    rows = []
    for lab, p in tags.items():
        if p.max() >= thr:
            rows.append({"label": lab, "confiance_moy": round(float(p.mean()), 3), "confiance_max": round(float(p.max()), 3),
                         "pct_temps": round(float((p >= thr).mean() * 100), 1)})
    return sorted(rows, key=lambda r: -r["confiance_moy"])


def niveau(r):
    if r["pct_temps"] >= 60 and r["confiance_moy"] >= 0.25: return "élevé"
    if r["pct_temps"] >= 30 and r["confiance_moy"] >= 0.10: return "moyen"
    return "faible"


def fmt_dur(s):
    return f"{int(s // 60)} min {int(s % 60):02d} s ({s:.1f} s)"


def write_report(name, meta, res, thr, dest):
    comment = meta["tags"].get("comment", "")
    L = [f"# Rapport : {name}", "", f"_Généré le {datetime.datetime.now():%Y-%m-%d %H:%M}, seuil de confiance {thr}_", "",
         "## Informations du fichier", "", f"- **Durée** : {fmt_dur(meta['duree_s'])}",
         f"- Codec : {meta['codec']}, {meta['echantillonnage_hz']} Hz, {meta['canaux']} canaux, débit {meta['debit_bps']} bit/s",
         f"- Taille : {meta['taille_octets'] / 1e6:.1f} Mo", f"- **Commentaire (comment)** : {comment or '(vide)'}",
         "- Métadonnées complètes (ffprobe) :"]
    L += [f"  - `{k}` : {v}" for k, v in meta["tags"].items()] or ["  - (aucune)"]
    L += ["", "## Instruments détectés (synthèse)", "",
          "| Instrument | Niveau | Conf. moy. | Conf. max | % du temps | Source |", "|---|---|---|---|---|---|"]
    for r in res["synthese"]:
        L.append(f"| {r['label']} | {r['niveau']} | {r['confiance_moy']} | {r['confiance_max']} | {r['pct_temps']} | {r['source']} |")
    L += ["", "_Source = mix ou piste Demucs où le score est le meilleur. Les pistes séparées contiennent des artefacts : "
          "un instrument vu seulement sur une piste isolée est moins sûr._"]
    for src, rows in res["par_source"].items():
        L += ["", f"## Détail — {src}", "", "| Label | Conf. moy. | Conf. max | % du temps |", "|---|---|---|---|"]
        L += [f"| {r['label']} | {r['confiance_moy']} | {r['confiance_max']} | {r['pct_temps']} |" for r in rows] or ["| (rien au-dessus du seuil) | | | |"]
    (dest / "rapport.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (dest / "rapport.json").write_text(json.dumps({"morceau": name, "metadonnees": meta, **res}, ensure_ascii=False, indent=2), encoding="utf-8")


def analyse_track(audio_file, out, model, labels, device, thr, progress=print, lyrics=True, lyrics_model="large-v3"):
    """Analyse un fichier audio complet, écrit rapport.md/json dans out/<nom>/ et retourne (meta, synthèse)."""
    audio_file = Path(audio_file)
    dest = Path(out) / audio_file.stem
    dest.mkdir(parents=True, exist_ok=True)
    meta = ffprobe(audio_file)
    progress("Séparation des pistes (Demucs)…")
    stems = separate(audio_file, dest, device)
    sources = {"mix": audio_file, **{s: stems / f"{s}.wav" for s in STEMS}}
    par_source, best = {}, {}
    for src, path in sources.items():
        progress(f"Détection des instruments : {src}…")
        audio, _ = librosa.load(path, sr=SR, mono=True)
        rows = summarize(tag(model, labels, audio), thr)
        par_source[src] = rows
        for r in rows:
            if r["label"] not in best or r["confiance_moy"] > best[r["label"]]["confiance_moy"]:
                best[r["label"]] = {**r, "source": src}
    synth = sorted(best.values(), key=lambda r: -r["confiance_moy"])
    for r in synth:
        r["niveau"] = niveau(r)
    res = {"seuil": thr, "synthese": synth, "par_source": par_source}
    write_report(audio_file.stem, meta, res, thr, dest)
    lyr = dest / "paroles.txt"
    if lyrics:
        import paroles
        if paroles.has_voice(res):
            progress("Transcription des paroles (Whisper)…")
            try:
                paroles.transcribe(stems / "vocals.wav", dest, lyrics_model)
            except Exception as e:  # les paroles sont un bonus : l'analyse reste valide
                progress(f"Paroles non générées : {e}")
        else:
            lyr.unlink(missing_ok=True)
            progress("Pas de voix détectée : pas de paroles.")
    return meta, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "input"))
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--seuil", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sans-paroles", action="store_true", help="ne pas transcrire les paroles (Whisper)")
    ap.add_argument("--modele-paroles", default="large-v3", help="tiny, base, small, medium, large-v3")
    a = ap.parse_args()
    import torch
    from panns_inference import AudioTagging, labels
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique : {device}")
    mp3s = sorted(p for p in Path(a.input).iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not mp3s:
        sys.exit(f"Aucun MP3/WAV dans {a.input}")
    model = AudioTagging(checkpoint_path=None, device=device)
    out = Path(a.output); out.mkdir(exist_ok=True)
    all_res = {}
    for mp3 in mp3s:
        name = mp3.stem
        print(f"\n=== {name} ===")
        meta, synth = analyse_track(mp3, out, model, labels, device, a.seuil, lyrics=not a.sans_paroles, lyrics_model=a.modele_paroles)
        all_res[name] = (meta, synth)
        print(", ".join(f"{r['label']} ({r['niveau']})" for r in synth[:8]))
    # synthèse globale
    labs = sorted({r["label"] for _, s in all_res.values() for r in s},
                  key=lambda l: -sum(r["confiance_moy"] for _, s in all_res.values() for r in s if r["label"] == l))
    names = list(all_res)
    L = ["# Synthèse globale", "", f"Seuil {a.seuil}. Cellule = niveau (confiance moyenne, source).", "",
         "| Morceau | Durée | Comment |", "|---|---|---|"]
    L += [f"| {n} | {fmt_dur(all_res[n][0]['duree_s'])} | {all_res[n][0]['tags'].get('comment', '') or '(vide)'} |" for n in names]
    L += ["", "| Instrument | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for l in labs:
        cells = []
        for n in names:
            r = next((r for r in all_res[n][1] if r["label"] == l), None)
            cells.append(f"{r['niveau']} ({r['confiance_moy']}, {r['source']})" if r else "—")
        L.append(f"| {l} | " + " | ".join(cells) + " |")
    (out / "synthese.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nTerminé. Voir {out}/synthese.md")


if __name__ == "__main__":
    main()
