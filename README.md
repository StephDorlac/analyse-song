# analyse-song

Outil local d'analyse de morceaux (MP3 / WAV) avec interface web :

1. **Analyse des instruments** : séparation en 4 pistes (voix, batterie, basse, autres) avec Demucs, puis reconnaissance des instruments avec PANNs (étiquettes AudioSet). Transcription des **paroles** avec Whisper.
2. **Génération MIDI** : un fichier `.mid` par morceau (voix, basse, autres, batterie), tempo et mesure détectés (4/4 ou 12/8).
3. **Partition** : affichage sur portées, avec **paroles** sous les notes, **accords** en notation US (C, Dm, G7, Fmaj7…) et **lecture audio** synchronisée (barre rouge, pages qui tournent).
4. **Métadonnées** : voir, modifier ou supprimer les tags (titre, artiste, commentaire…) dans une copie du fichier, sans ré-encodage.

Tout tourne sur ta machine : aucun fichier n'est envoyé sur Internet (hors téléchargement des modèles à la première utilisation).

## Prérequis

- Linux (testé sur Linux Mint 22 / Ubuntu 24.04), Python 3.12
- `ffmpeg` : `sudo apt install ffmpeg`
- Pour le rendu audio de contrôle du MIDI (facultatif) : `sudo apt install fluidsynth fluid-soundfont-gm`
- Un GPU NVIDIA accélère beaucoup l'analyse (facultatif ; sinon CPU, plus lent)

## Installation

```bash
git clone https://github.com/StephDorlac/analyse-song.git
cd analyse-song
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Les modèles (Demucs, PANNs, Whisper `large-v3`) sont téléchargés automatiquement au premier lancement : prévoir quelques Go et du temps.

## Utilisation

### Interface web (recommandé)

```bash
./ui.sh
```

Le navigateur s'ouvre sur http://127.0.0.1:5000 (autre port : `PORT=5050 ./ui.sh`).

| Onglet | Que faire |
|---|---|
| **Analyse des instruments** | Glisser des MP3/WAV, cocher les morceaux, régler le *seuil de confiance*, « Analyser la sélection ». Le bouton **▶ Écouter** joue le morceau, **Voir** affiche le rapport (instruments, paroles, pistes séparées à écouter) ; chaque piste séparée (voix, batterie, basse, autres) se télécharge en **wav ou mp3**, ou toutes ensemble en zip. |
| **Génération MIDI** | Choisir des morceaux déjà analysés, le timing (quantifié ou brut), l'instrument de la piste « autres », puis « Générer ». Télécharger le `.mid`. |
| **Partition** | Choisir le morceau, le nombre de mesures par page, la taille, les portées affichées. Le lecteur joue le morceau d'origine : la barre rouge suit la position et les pages tournent automatiquement (décocher pour garder la page fixe). |
| **Métadonnées** | Modifier/supprimer des champs, « Enregistrer une copie corrigée » puis la télécharger. Le fichier d'origine n'est jamais modifié. |

### Ligne de commande

```bash
./analyse.sh                       # analyse tous les fichiers de input/ → output/
./analyse.sh --seuil 0.15          # seuil de confiance (défaut 0.10)
./analyse.sh --sans-paroles        # sans transcription Whisper
./analyse.sh --modele-paroles small  # modèle Whisper plus léger (tiny, base, small, medium, large-v3)

./midi.sh                          # MIDI de tous les morceaux déjà analysés
./midi.sh --morceau "Nom"          # un seul morceau
./midi.sh --version brut           # timing fidèle à l'interprétation (défaut : quantifié)
./midi.sh --sans-controle          # pas de rendu audio de contrôle
./midi.sh --other-program 27       # instrument GM imposé pour la piste « autres »
```

### Où sont les résultats ?

```
input/                 tes MP3 / WAV (jamais modifiés)
output/<morceau>/
  rapport.md / .json   instruments détectés (confiance, % du temps, piste)
  paroles.txt / .json  paroles, avec horodatage par mot
  stems/               voix, batterie, basse, autres (WAV)
  midi/                <morceau>.mid, controle.wav, partition.musicxml
  metadata_edit/       copie avec métadonnées corrigées
output/synthese.md     comparaison de tous les morceaux
```

## Comprendre les résultats

- **Seuil de confiance** : score minimal (0 à 1) pour retenir un instrument. Plus bas = plus d'instruments mais plus de faux positifs ; plus haut = liste courte et fiable mais un instrument discret peut disparaître. En musique, 0,1 à 0,3 est déjà un signal.
- **MIDI et partition = aide à la lecture**, pas une partition d'éditeur : tempo et mesure sont fiables, basse et batterie plutôt bonnes, voix et « autres » approximatives (notes parasites possibles).
- **Accords** : devinés temps par temps à partir des pistes « autres » et basse ; ils peuvent se tromper (inversions, accords enrichis).
- **Paroles** : transcription automatique à vérifier ; chaque mot est posé sous la note de voix la plus proche de son horodatage, certains mots peuvent manquer ou être décalés.
- La lecture synchronisée s'appuie sur la carte de tempo détectée : un léger décalage est possible si le tempo a été mal détecté.

## Dépannage

| Problème | Solution |
|---|---|
| `Venv absent` | Refaire l'installation (`python3 -m venv .venv` puis `pip install -r requirements.txt`). |
| `ffmpeg manquant` / `fluidsynth manquant` | `sudo apt install ffmpeg fluidsynth fluid-soundfont-gm` |
| Mémoire GPU insuffisante | `./analyse.sh --device cpu`, ou un modèle Whisper plus petit (`--modele-paroles small`). |
| Une modification Python n'apparaît pas | Relancer `./ui.sh` (le HTML se recharge seul, pas le Python). |
| Pas de barre rouge / partition vide | Générer d'abord le MIDI du morceau (onglet « Génération MIDI »). |
| Première ouverture d'une partition lente | Normal : Whisper refait la transcription avec horodatage par mot (≈ 1 min) pour les morceaux analysés avant cette fonction. |

## Licences et crédits

Code du projet : [MIT](LICENSE). Bibliothèques tierces : Demucs, PANNs, Whisper, basic-pitch, music21, Flask… (voir leurs licences respectives). La partition est affichée avec [OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/) (licence BSD-3, voir `static/OSMD-LICENSE.txt`).

Respecte les droits d'auteur des morceaux que tu analyses : les fichiers audio et les résultats sont exclus du dépôt (`.gitignore`).
