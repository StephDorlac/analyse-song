"""Transcription des paroles (Whisper) à partir de la piste voix séparée par Demucs."""
import argparse, json, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Phrases que Whisper invente sur du silence ou de la musique (fin de morceau surtout)
HALLUCINATIONS = {"thank you", "thank you.", "thanks for watching", "thanks for watching!", "merci.", "merci d'avoir regardé cette vidéo.",
                  "sous-titrage st' 501", "you"}
VOICE_LABELS = {"Singing", "Male singing", "Female singing", "Child singing", "Choir", "Speech", "Synthetic singing"}


def has_voice(rapport):
    """Vrai si PANNs a détecté du chant/de la parole sur la piste voix (au moins 15 % du temps)."""
    rows = rapport.get("par_source", {}).get("vocals", [])
    return any(r["label"] in VOICE_LABELS and r["pct_temps"] >= 15 for r in rows)


def load_model(name, device):
    import whisper
    m = whisper.load_model(name, device="cpu")
    if device == "cuda":  # demi-précision : 2x moins de mémoire GPU (les LayerNorm restent en pleine précision)
        import torch
        m = m.half()
        for mod in m.modules():
            if isinstance(mod, torch.nn.LayerNorm): mod.float()
    return m.to(device)


def free_gpu():
    """À appeler APRÈS avoir supprimé toutes les références au modèle."""
    import gc, torch, whisper.audio
    whisper.audio.mel_filters.cache_clear()  # whisper garde ce tableau en cache sur le GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe(vocals, dest, model_name="large-v3", language=None, device=None):
    """Écrit paroles.txt (texte seul) et paroles.json (segments horodatés). Retourne le dict résultat.
    Le modèle est toujours déchargé à la fin, même en cas d'erreur (bloc finally)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        model = load_model(model_name, device)
        r = model.transcribe(str(vocals), language=language, task="transcribe", fp16=(device == "cuda"),
                             condition_on_previous_text=False,   # évite que Whisper boucle sur la même phrase
                             no_speech_threshold=0.5, compression_ratio_threshold=2.0, temperature=(0.0, 0.2, 0.4),
                             word_timestamps=True)  # horodatage par mot : sert à placer les paroles sous les notes de la partition
    except BaseException as e:
        traceback.clear_frames(e.__traceback__)  # la trace d'erreur garderait sinon le modèle en mémoire
        raise
    finally:
        model = None    # toujours exécuté : succès, erreur ou interruption (Ctrl+C)
        free_gpu()
    r["segments"] = [x for x in r["segments"] if x["text"].strip().lower() not in HALLUCINATIONS]
    segs = [{"debut": round(s["start"], 2), "fin": round(s["end"], 2), "texte": s["text"].strip(),
             "no_speech": round(float(s["no_speech_prob"]), 2),
             "mots": [{"m": w["word"].strip(), "debut": round(w["start"], 2), "fin": round(w["end"], 2)} for w in s.get("words", []) if w["word"].strip()]}
            for s in r["segments"] if s["text"].strip()]
    dest = Path(dest)
    (dest / "paroles.txt").write_text("\n".join(s["texte"] for s in segs) + "\n", encoding="utf-8")
    res = {"langue": r["language"], "modele": model_name, "segments": segs}
    (dest / "paroles.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def ensure_words(dest):
    """Retourne les segments de paroles.json avec leurs mots horodatés ; relance Whisper si le fichier date d'avant cette option.
    Retourne None s'il n'y a pas de paroles."""
    dest = Path(dest)
    f = dest / "paroles.json"
    if not f.exists():
        return None
    res = json.loads(f.read_text(encoding="utf-8"))
    if res["segments"] and not all("mots" in sg for sg in res["segments"]):
        res = transcribe(dest / "stems" / "vocals.wav", dest, res.get("modele", "large-v3"), res.get("langue"))
    return res["segments"]


def main():
    ap = argparse.ArgumentParser(description="Paroles d'un morceau déjà analysé (output/<morceau>/stems/vocals.wav)")
    ap.add_argument("morceau", nargs="?", help="nom du dossier dans output/ (tous si absent)")
    ap.add_argument("--modele", default="large-v3", help="tiny, base, small, medium, large-v3")
    ap.add_argument("--langue", default=None, help="en, fr… (auto par défaut)")
    a = ap.parse_args()
    out = ROOT / "output"
    for d in sorted(p for p in out.iterdir() if p.is_dir() and (a.morceau is None or p.name == a.morceau)):
        rap = d / "rapport.json"
        if not rap.exists() or not (d / "stems" / "vocals.wav").exists():
            print(f"{d.name} : analyse manquante"); continue
        if not has_voice(json.loads(rap.read_text(encoding="utf-8"))):
            print(f"{d.name} : pas de voix détectée, ignoré"); continue
        print(f"{d.name} : transcription…")
        r = transcribe(d / "stems" / "vocals.wav", d, a.modele, a.langue)
        print(f"  langue {r['langue']}, {len(r['segments'])} lignes -> {d / 'paroles.txt'}")


if __name__ == "__main__":
    main()
