"""Interface web locale (http://127.0.0.1:5000). Lancée par ui.sh."""
import json, os, re, shutil, subprocess, threading, traceback
from types import SimpleNamespace
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory, abort

import analyse
import midi
import paroles
import score

ROOT = Path(__file__).resolve().parent
INPUT, OUTPUT = ROOT / "input", ROOT / "output"
INPUT.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True  # le HTML modifié est pris en compte sans relancer le serveur

state = {"kind": None, "running": False, "queue": [], "current": None, "message": "", "done": [], "errors": {}}
lock = threading.Lock()
_model = {}


def audio_files():
    return sorted((p for p in INPUT.iterdir() if p.suffix.lower() in analyse.AUDIO_EXT), key=lambda p: p.name.lower())


def get_model():
    if not _model:
        import torch
        from panns_inference import AudioTagging, labels
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.update(device=device, labels=labels, model=AudioTagging(checkpoint_path=None, device=device))
    return _model


def worker(kind, names, opts):
    try:
        for name in names:
            with lock:
                state["current"] = name
                state["message"] = "Chargement du modèle…" if kind == "analyse" else "Transcription (voix, basse, autres, batterie)…"
            try:
                if kind == "analyse":
                    m = get_model()
                    analyse.analyse_track(INPUT / name, OUTPUT, m["model"], m["labels"], m["device"], opts["seuil"],
                                          progress=lambda msg: state.update(message=msg), lyrics=opts.get("paroles", True))
                else:
                    args = SimpleNamespace(extrait=None, other_program=opts.get("other_program"),
                                           version=opts["version"], controle=opts["controle"])
                    if midi.process(name, OUTPUT, args) is None:
                        raise RuntimeError("analyse préalable manquante")
                with lock:
                    state["done"].append(name)
            except Exception as e:  # une erreur ne bloque pas les morceaux suivants
                traceback.print_exc()
                with lock:
                    state["errors"][name] = str(e)
    finally:
        with lock:
            state.update(running=False, current=None, message="Terminé")


def start_job(kind, names, opts):
    with lock:
        if state["running"] or not names:
            return jsonify(error="Un traitement est déjà en cours ou aucun morceau sélectionné"), 409
        state.update(kind=kind, running=True, queue=names, done=[], errors={}, message="Démarrage…")
    threading.Thread(target=worker, args=(kind, names, opts), daemon=True).start()
    return jsonify(ok=True)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/tracks")
def tracks():
    return jsonify([{"name": p.name, "stem": p.stem, "size": p.stat().st_size,
                     "analysed": (OUTPUT / p.stem / "rapport.json").exists()} for p in audio_files()])


@app.get("/api/audio/<name>")
def audio_play(name):
    """Sert l'audio d'origine en lecture (flux avec Range, pas de téléchargement forcé)."""
    f = INPUT / Path(name).name
    if f.suffix.lower() not in analyse.AUDIO_EXT or not f.exists():
        abort(404)
    return send_from_directory(INPUT, f.name)


@app.get("/api/stem/<stem>/<piste>")
def stem_play(stem, piste):
    """Sert une piste Demucs (vocals / drums / bass / other) en lecture."""
    if piste not in ("vocals", "drums", "bass", "other"):
        abort(404)
    return send_from_directory(OUTPUT / Path(stem).name / "stems", f"{piste}.wav")


@app.post("/api/upload")
def upload():
    saved = []
    for f in request.files.getlist("files"):
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in analyse.AUDIO_EXT:
            continue
        dest, n = INPUT / name, 2
        while dest.exists():  # on n'écrase jamais un fichier existant
            dest = INPUT / f"{Path(name).stem} ({n}){Path(name).suffix}"; n += 1
        f.save(dest)
        saved.append(dest.name)
    return jsonify(saved)


@app.post("/api/analyse")
def start():
    data = request.get_json()
    valid = {p.name for p in audio_files()}
    names = [n for n in data.get("files", []) if n in valid]
    return start_job("analyse", names, {"seuil": float(data.get("seuil", 0.10)), "paroles": bool(data.get("paroles", True))})


# ---------- MIDI (séparé de l'analyse) ----------
def midi_dir(stem):
    return OUTPUT / Path(stem).name / "midi"


@app.get("/api/midi/tracks")
def midi_tracks():
    rows = []
    for p in audio_files():
        if not (OUTPUT / p.stem / "rapport.json").exists():
            continue  # le MIDI part des pistes issues de l'analyse
        d = midi_dir(p.stem)
        rows.append({"name": p.name, "stem": p.stem, "midi": (d / f"{p.stem}.mid").exists(), "controle": (d / "controle.wav").exists()})
    return jsonify(rows)


@app.post("/api/midi/generate")
def midi_generate():
    data = request.get_json()
    valid = {p.name for p in audio_files() if (OUTPUT / p.stem / "rapport.json").exists()}
    names = [n for n in data.get("files", []) if n in valid]
    opts = {"version": "brut" if data.get("version") == "brut" else "quantifie", "controle": bool(data.get("controle", True))}
    if data.get("other_program") not in (None, ""):
        opts["other_program"] = int(data["other_program"])
    return start_job("midi", [Path(n).stem for n in names], opts)


@app.get("/api/score/<stem>")
def score_info(stem):
    """Construit (si besoin) la partition MusicXML à partir du .mid, puis retourne ses infos."""
    stem = Path(stem).name
    d = midi_dir(stem)
    mid, xml, js = d / f"{stem}.mid", d / "partition.musicxml", d / "partition.json"
    if not mid.exists():
        abort(404)
    lyr = OUTPUT / stem / "paroles.json"
    stale = not xml.exists() or not js.exists() or xml.stat().st_mtime < mid.stat().st_mtime
    if not stale:
        stale = "grille" not in json.loads(js.read_text(encoding="utf-8")) or (lyr.exists() and xml.stat().st_mtime < lyr.stat().st_mtime)
    if stale:
        with lock:
            if state["running"]:
                return jsonify(error="Un traitement est en cours (GPU occupé), réessayez après"), 409
        rep = d / "rapport_midi.json"
        info = json.loads(rep.read_text(encoding="utf-8"))
        if "lead" not in info:  # MIDI généré avant la sauvegarde du décalage MIDI/audio
            info["lead"] = midi.compute_lead(stem, OUTPUT)
            rep.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        res = score.build_score(mid, xml, info, paroles.ensure_words(OUTPUT / stem))
        res["tempo"] = info["tempo"]
        js.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return jsonify(json.loads(js.read_text(encoding="utf-8")))


@app.get("/api/score/<stem>/xml")
def score_xml(stem):
    stem = Path(stem).name
    return send_from_directory(midi_dir(stem), "partition.musicxml", mimetype="application/vnd.recordare.musicxml+xml")


@app.get("/api/midi/<stem>/report")
def midi_report(stem):
    f = midi_dir(stem) / "rapport_midi.json"
    if not f.exists():
        abort(404)
    return jsonify(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/midi/<stem>/file")
def midi_file(stem):
    stem = Path(stem).name
    return send_from_directory(midi_dir(stem), f"{stem}.mid", as_attachment=True, download_name=f"{stem}.mid")


@app.get("/api/midi/<stem>/audio")
def midi_audio(stem):
    return send_from_directory(midi_dir(stem), "controle.wav")


@app.delete("/api/midi/<stem>")
def midi_delete(stem):
    with lock:
        if state["running"]:
            return jsonify(error="Un traitement est en cours, réessayez après"), 409
    shutil.rmtree(midi_dir(stem), ignore_errors=True)
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    with lock:
        return jsonify(state)


@app.get("/api/report/<stem>")
def report(stem):
    f = OUTPUT / Path(stem).name / "rapport.json"
    if not f.exists():
        abort(404)
    return jsonify(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/lyrics/<stem>")
def lyrics(stem):
    f = OUTPUT / Path(stem).name / "paroles.txt"
    if not f.exists():
        abort(404)
    return jsonify(texte=f.read_text(encoding="utf-8"))


@app.get("/api/lyrics/<stem>/file")
def lyrics_file(stem):
    stem = Path(stem).name
    return send_from_directory(OUTPUT / stem, "paroles.txt", as_attachment=True, download_name=f"{stem} - paroles.txt")


@app.get("/api/report/<stem>/md")
def report_md(stem):
    return send_from_directory(OUTPUT / Path(stem).name, "rapport.md", as_attachment=True, download_name=f"{Path(stem).name}.md")


@app.delete("/api/track/<name>")
def delete_track(name):
    """Supprime l'analyse (output/<nom>/) ; avec ?audio=1, supprime aussi le fichier de input/."""
    name = Path(name).name
    with lock:
        if state["running"]:
            return jsonify(error="Analyse en cours, réessayez après"), 409
    f = INPUT / name
    if f.suffix.lower() not in analyse.AUDIO_EXT or not f.exists():
        abort(404)
    shutil.rmtree(OUTPUT / f.stem, ignore_errors=True)
    if request.args.get("audio") == "1":
        f.unlink()
    return jsonify(ok=True)


# ---------- Métadonnées (l'original de input/ n'est jamais modifié : on écrit une copie dans output/) ----------
KEY_RE = re.compile(r"^[A-Za-z0-9_ .-]{1,40}$")


def audio_by_name(name):
    f = INPUT / Path(name).name
    if f.suffix.lower() not in analyse.AUDIO_EXT or not f.exists():
        abort(404)
    return f


def edited_path(f):
    return OUTPUT / f.stem / "metadata_edit" / f.name


def meta_info(path):
    m = analyse.ffprobe(path)
    return {"tags": m["tags"], "duree_s": m["duree_s"], "codec": m["codec"], "echantillonnage_hz": m["echantillonnage_hz"],
            "canaux": m["canaux"], "taille_octets": m["taille_octets"]}


@app.get("/api/meta/<name>")
def meta_get(name):
    f = audio_by_name(name)
    e = edited_path(f)
    return jsonify(original=meta_info(f), edited=meta_info(e) if e.exists() else None, format=f.suffix.lower().lstrip("."))


@app.post("/api/meta/<name>")
def meta_save(name):
    f = audio_by_name(name)
    tags = request.get_json().get("tags", {})
    for k, v in tags.items():
        if not KEY_RE.match(k) or "\n" in str(v) or "\x00" in str(v):
            return jsonify(error=f"Nom de champ ou valeur invalide : {k!r}"), 400
    dest = edited_path(f)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(f), "-map", "0", "-c", "copy", "-map_metadata", "-1"]  # copie sans ré-encodage, métadonnées vidées
    if f.suffix.lower() == ".mp3":
        cmd += ["-id3v2_version", "3"]
    for k, v in tags.items():
        if str(v).strip():
            cmd += ["-metadata", f"{k}={v}"]
    tmp = dest.with_name("tmp_" + dest.name)  # écriture dans un fichier temporaire puis renommage
    r = subprocess.run(cmd + [str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return jsonify(error=r.stderr.strip()[-300:] or "ffmpeg a échoué"), 500
    tmp.replace(dest)
    return jsonify(edited=meta_info(dest))


@app.get("/api/meta/<name>/download")
def meta_download(name):
    f = audio_by_name(name)
    e = edited_path(f)
    if not e.exists():
        abort(404)
    return send_from_directory(e.parent, e.name, as_attachment=True)


@app.delete("/api/meta/<name>/edited")
def meta_delete_copy(name):
    shutil.rmtree(edited_path(audio_by_name(name)).parent, ignore_errors=True)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
