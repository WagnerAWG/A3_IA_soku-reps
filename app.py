import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory, abort

from game_data import CHARACTERS
from deck_optimizer import DeckOptimizer

app = Flask(__name__, static_folder="static", template_folder="templates")

VISUALS_CHARACTER_DIR = Path(__file__).parent / "Visuais" / "09b_character"
VISUALS_SELECT_DIR = Path(__file__).parent / "Visuais" / "05a_selcha"
VISUALS_CARD_DIR = Path(__file__).parent / "Visuais" / "card"
DATA_FILE = Path(__file__).parent / "data.csv"

CHARACTER_IMAGE_FOLDERS = {
    0: "reimu",
    1: "marisa",
    2: "sakuya",
    3: "alice",
    4: "patchouli",
    5: "youmu",
    6: "remilia",
    7: "yuyuko",
    8: "yukari",
    9: "suika",
    10: "Reisen",
    11: "aya",
    12: "komachi",
    13: "iku",
    14: "tenshi",
    15: "sanae",
    16: "Cirno",
    17: "Meiling",
    18: "utsuho",
    19: "suwako",
}

MODEL = None
MODEL_TYPE = "fallback"


def _find_image_file(directory: Path, pattern: str):
    if not directory.exists():
        return None, None

    matches = sorted(directory.glob(pattern))
    if not matches:
        return None, None

    return directory, matches[0].name


def _get_character_image_filename(char_id):
    if char_id is None or char_id < 0 or char_id > 19:
        return None, None
    return _find_image_file(VISUALS_CHARACTER_DIR, f"character_{char_id:02d}_*.png")


def _get_character_splash_filename(char_id):
    if char_id is None or char_id < 0 or char_id > 19:
        return None, None
    return _find_image_file(VISUALS_SELECT_DIR, f"selcha_{char_id:02d}_*.png")


def _get_character_card_files(char_id, limit=8):
    folder_name = CHARACTER_IMAGE_FOLDERS.get(char_id)
    if not folder_name:
        return []

    character_dir = VISUALS_CARD_DIR / folder_name
    if not character_dir.exists():
        return []

    cards = []
    for group in ["skill", "spell"]:
        group_dir = character_dir / group
        if not group_dir.exists():
            continue

        for file in sorted(group_dir.glob("*.png")):
            cards.append({
                "type": group,
                "filename": file.name,
                "url": f"/card-image/{char_id}/{group}/{file.name}",
                "label": file.stem.replace("card", "").replace("_", " ").strip(),
            })
            if len(cards) >= limit:
                break
        if len(cards) >= limit:
            break

    return cards


def _train_model():
    global MODEL, MODEL_TYPE
    if not DATA_FILE.exists():
        return

    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        dataset = pd.read_csv(DATA_FILE)
        required = ["server_rank", "client_rank", "server_char", "client_char", "winner"]
        if not all(col in dataset.columns for col in required):
            return

        X = dataset[["server_rank", "client_rank", "server_char", "client_char"]].astype(int)
        y = dataset["winner"].apply(lambda value: 1 if str(value).strip().lower() == "server" else 0)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        MODEL = model
        MODEL_TYPE = "trained"
    except Exception:
        MODEL = None
        MODEL_TYPE = "fallback"


def _predict_with_model(server_rank, client_rank, server_char, client_char):
    if MODEL is None:
        return None

    import numpy as np

    X = [[server_rank, client_rank, server_char, client_char]]
    proba = MODEL.predict_proba(X)[0][1]
    proba = float(max(min(proba, 0.95), 0.05))
    return {
        "winner": "server" if proba >= 0.5 else "client",
        "server_win_probability": proba,
        "client_win_probability": 1.0 - proba,
        "model": MODEL_TYPE,
    }


def _fallback_predict(server_rank, client_rank, server_char, client_char):
    diff = float(server_rank - client_rank)
    score = 0.5 + max(min(diff / 1000.0, 0.2), -0.2)
    score += (client_char - server_char) * 0.003
    score = min(max(score, 0.06), 0.94)

    return {
        "winner": "server" if score >= 0.5 else "client",
        "server_win_probability": score,
        "client_win_probability": 1.0 - score,
        "model": MODEL_TYPE,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/character-image/<int:char_id>")
def character_image(char_id):
    folder, filename = _get_character_image_filename(char_id)
    if not folder or not filename:
        abort(404)
    return send_from_directory(str(folder), filename)


@app.route("/select-splash/<int:char_id>")
def select_splash(char_id):
    folder, filename = _get_character_splash_filename(char_id)
    if not folder or not filename:
        abort(404)
    return send_from_directory(str(folder), filename)


@app.route("/card-image/<int:char_id>/<card_type>/<path:filename>")
def card_image(char_id, card_type, filename):
    folder_name = CHARACTER_IMAGE_FOLDERS.get(char_id)
    if not folder_name or card_type not in {"skill", "spell"}:
        abort(404)

    folder = VISUALS_CARD_DIR / folder_name / card_type
    if not folder.exists():
        abort(404)

    return send_from_directory(str(folder), filename)


@app.route("/api/characters", methods=["GET"])
def api_characters():
    return jsonify([{"id": cid, "name": name} for cid, name in CHARACTERS.items()])


@app.route("/api/character-cards/<int:char_id>", methods=["GET"])
def api_character_cards(char_id):
    return jsonify(_get_character_card_files(char_id, limit=10))


@app.route("/api/optimized-deck/<int:char_id>", methods=["GET"])
def api_optimized_deck(char_id):
    if char_id not in CHARACTERS:
        return jsonify({"error": "ID de personagem inválido."}), 400

    if DECK_OPTIMIZER.model is None:
        return jsonify({"error": "Modelo de otimização de baralho não está disponível."}), 500

    deck, win_rate = DECK_OPTIMIZER.optimize_deck(char_id)
    if deck is None:
        return jsonify({"error": "Não foi possível gerar o baralho otimizado."}), 500

    return jsonify({
        "character": CHARACTERS[char_id],
        "estimated_win_probability": win_rate,
        "max_cards": 20,
        "max_copies": 4,
        "deck": DECK_OPTIMIZER.format_deck(char_id, deck),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body esperado"}), 400

    required = ["server_rank", "client_rank", "server_char", "client_char"]
    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({"error": f"Campos faltando: {', '.join(missing)}"}), 400

    try:
        server_rank = int(data["server_rank"])
        client_rank = int(data["client_rank"])
        server_char = int(data["server_char"])
        client_char = int(data["client_char"])
    except (ValueError, TypeError):
        return jsonify({"error": "Valores inválidos. ELO e IDs devem ser inteiros."}), 400

    if server_char not in CHARACTERS or client_char not in CHARACTERS:
        return jsonify({"error": "ID de personagem inválido."}), 400

    if MODEL is None and DATA_FILE.exists():
        _train_model()

    prediction = _predict_with_model(server_rank, client_rank, server_char, client_char)
    if prediction is None:
        prediction = _fallback_predict(server_rank, client_rank, server_char, client_char)

    return jsonify(prediction)


DECK_OPTIMIZER = DeckOptimizer()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
