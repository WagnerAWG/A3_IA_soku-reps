import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory, abort

from game_data import (
    CHARACTERS, SYSTEM_CARDS, CHAR_SKILL_CARDS, CHAR_SPELL_CARDS, CHAR_CARDS,
    get_card_name, get_spell_cost, get_skill_input, CHAR_SKILL_INPUTS,
)
from Baralho_Optimal import predict_match as predict_fn, random_deck as deck_random_deck
import joblib

app = Flask(__name__, static_folder="static", template_folder="templates")

character_strength = {}
_csp = Path(__file__).parent / "models" / "character_strength.pkl"
if _csp.exists():
    character_strength = joblib.load(_csp)

VISUALS_CHARACTER_DIR = Path(__file__).parent / "Visuais" / "09b_character"
VISUALS_SELECT_DIR = Path(__file__).parent / "Visuais" / "05a_selcha"
VISUALS_CARD_DIR = Path(__file__).parent / "Visuais" / "card"

CHARACTER_IMAGE_FOLDERS = {
    0: "reimu", 1: "marisa", 2: "sakuya", 3: "alice", 4: "patchouli",
    5: "youmu", 6: "remilia", 7: "yuyuko", 8: "yukari", 9: "suika",
    10: "Reisen", 11: "aya", 12: "komachi", 13: "iku", 14: "tenshi",
    15: "sanae", 16: "Cirno", 17: "Meiling", 18: "utsuho", 19: "suwako",
}

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


def _get_card_categories(char_id):
    system = []
    skills_by_input = {}
    spells_by_cost = {}

    for card_id in sorted(CHAR_CARDS.get(char_id, set())):
        name = get_card_name(char_id, card_id)
        safe = name.replace(' ', '_').replace('"', '')
        if card_id <= 20:
            img = f"/card-image/common/card{card_id:03d}_{safe}.png"
            system.append({"card_id": card_id, "name": name, "image_url": img})
        elif 100 <= card_id <= 114:
            inp = get_skill_input(char_id, card_id)
            img = f"/card-image/{char_id}/skill/card{card_id}_{safe}.png"
            skills_by_input.setdefault(inp, []).append({"card_id": card_id, "name": name, "image_url": img})
        elif card_id >= 200:
            cost = get_spell_cost(char_id, card_id)
            img = f"/card-image/{char_id}/spell/card{card_id}_{safe}.png"
            spells_by_cost.setdefault(str(cost), []).append({"card_id": card_id, "name": name, "image_url": img})

    return {"system": system, "skills": skills_by_input, "spells": spells_by_cost}


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


@app.route("/card-image/common/<path:filename>")
def card_image_common(filename):
    folder = VISUALS_CARD_DIR / "common"
    if not folder.exists():
        abort(404)
    return send_from_directory(str(folder), filename)


@app.route("/api/characters", methods=["GET"])
def api_characters():
    return jsonify([{"id": cid, "name": name} for cid, name in CHARACTERS.items()])


@app.route("/api/character-cards/<int:char_id>", methods=["GET"])
def api_character_cards(char_id):
    return jsonify(_get_character_card_files(char_id, limit=10))


@app.route("/api/character-cards-all/<int:char_id>", methods=["GET"])
def api_character_cards_all(char_id):
    if char_id not in CHARACTERS:
        return jsonify({"error": "ID inválido"}), 400
    return jsonify(_get_card_categories(char_id))


@app.route("/api/optimized-deck/<int:char_id>", methods=["GET"])
def api_optimized_deck(char_id):
    if char_id not in CHARACTERS:
        return jsonify({"error": "ID de personagem inválido."}), 400
    if deck_random_deck is None:
        return jsonify({"error": "Modelo de baralho indisponível."}), 500

    from collections import Counter
    import random
    from Baralho_Optimal import (
        evaluate_population, random_deck, repair_deck, mutate,
        tournament_select, crossover,
    )

    POP_SIZE = 20
    GENERATIONS = 10
    ELITISM_COUNT = 3

    population = [random_deck(char_id) for _ in range(POP_SIZE)]
    for _ in range(GENERATIONS):
        scores = evaluate_population(char_id, population)
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        new_pop = [d for _, d in ranked[:ELITISM_COUNT]]
        while len(new_pop) < POP_SIZE:
            if random.random() < 0.7:
                p1 = tournament_select(population, scores)
                p2 = tournament_select(population, scores)
                child = crossover(p1, p2)
            else:
                child = tournament_select(population, scores)
            child = repair_deck(mutate(child, char_id), char_id)
            new_pop.append(child)
        population = new_pop

    final_scores = evaluate_population(char_id, population)
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    best_deck = population[best_idx]
    best_prob = float(final_scores[best_idx])

    counts = Counter(best_deck)
    formatted = [{"card_id": cid, "name": get_card_name(char_id, cid), "count": n}
                 for cid, n in sorted(counts.items())]

    return jsonify({
        "character": CHARACTERS[char_id],
        "estimated_win_probability": best_prob,
        "max_cards": 20,
        "max_copies": 4,
        "deck": formatted,
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
        server_cards = data.get("server_cards", [])
        client_cards = data.get("client_cards", [])
    except (ValueError, TypeError):
        return jsonify({"error": "Valores inválidos."}), 400

    if server_char not in CHARACTERS or client_char not in CHARACTERS:
        return jsonify({"error": "ID de personagem inválido."}), 400

    if predict_fn is not None and server_cards and client_cards:
        result = predict_fn(server_rank, client_rank, server_char, client_char,
                            server_cards, client_cards)
    elif predict_fn is not None:
        if not server_cards:
            server_cards = deck_random_deck(server_char)
        if not client_cards:
            client_cards = deck_random_deck(client_char)
        result = predict_fn(server_rank, client_rank, server_char, client_char,
                            server_cards, client_cards)
    else:
        return jsonify({"error": "Modelo indisponivel."}), 500

    ss = character_strength.get(server_char, 1.0)
    cs = character_strength.get(client_char, 1.0)
    if ss > cs * 1.005:
        matchup = "Favorece Servidor"
    elif cs > ss * 1.005:
        matchup = "Favorece Cliente"
    else:
        matchup = "Equilibrado"

    return jsonify({"winner": result["winner"],
                     "server_win_probability": result["server_win_probability"],
                     "client_win_probability": result["client_win_probability"],
                     "model": "deck",
                     "matchup": matchup})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
