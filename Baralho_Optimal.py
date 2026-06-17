import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from game_data import (
    CHARACTERS, CHAR_CARDS, get_card_name,
    INTERNAL_CARD_INDEX, N_FEATURES, SKILL_SLOT,
)
import random
import joblib
from pathlib import Path

random.seed(42)
np.random.seed(42)

# 1 - CARREGAR DADOS
print("Carregando dados...")
df = pd.read_csv('data.csv')

mask = (df['serverUserElo'] >= 1600) & (df['clientUserElo'] >= 1600)
df_filtered = df[mask].copy()
print(f"Partidas com ambos ELO >= 1600: {len(df_filtered)}")
print(f"Total de features (IDs internos): {N_FEATURES}")

df_filtered['winner_binary'] = (df_filtered['winner'] == -1).astype(int)

# Coleta de decks reais e medias de composicao por personagem
CHAR_REAL_DECKS = {cid: [] for cid in range(20)}
char_sys_total = {cid: 0 for cid in range(20)}
char_skill_total = {cid: 0 for cid in range(20)}
char_spell_total = {cid: 0 for cid in range(20)}
char_deck_count = {cid: 0 for cid in range(20)}
char_unique_total = {cid: 0 for cid in range(20)}
char_card_count = {cid: defaultdict(int) for cid in range(20)}
char_card_wins = {cid: defaultdict(int) for cid in range(20)}
char_wins = {cid: 0 for cid in range(20)}

for _, row in df_filtered.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
    c_cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]
    s_won = row['winner'] == -1

    for char_id, cards, won in [(s_char, s_cards, s_won), (c_char, c_cards, not s_won)]:
        CHAR_REAL_DECKS[char_id].append(cards)
        char_sys_total[char_id] += sum(1 for c in cards if c <= 20)
        char_skill_total[char_id] += sum(1 for c in cards if 100 <= c <= 114)
        char_spell_total[char_id] += sum(1 for c in cards if c >= 200)
        char_deck_count[char_id] += 1
        char_unique_total[char_id] += len(set(cards))
        if won:
            char_wins[char_id] += 1
        for card_id in set(cards):
            char_card_count[char_id][card_id] += 1
            if won:
                char_card_wins[char_id][card_id] += 1

CHAR_CARD_FREQ = {}
CHAR_CARD_WR_DIFF = {}
SYS_THRESHOLD = 1.0
SKILL_THRESHOLD = 2.0
FREQ_FLOOR = 0.01
FREQ_DIVISOR = 20.0
for cid in range(20):
    n_decks = char_deck_count[cid]
    total_wins = char_wins[cid]
    for card_id in CHAR_CARDS[cid]:
        count = char_card_count[cid].get(card_id, 0)
        freq = 100.0 * count / max(1, n_decks)
        CHAR_CARD_FREQ[(cid, card_id)] = freq

        card_w = char_card_wins[cid].get(card_id, 0)
        wr_with = card_w / max(1, count)
        decks_without = n_decks - count
        wins_without = total_wins - card_w
        wr_without = wins_without / max(1, decks_without)
        CHAR_CARD_WR_DIFF[(cid, card_id)] = wr_with - wr_without


def card_weight(char_id, card_id):
    freq = CHAR_CARD_FREQ.get((char_id, card_id), 0)
    wr_diff = CHAR_CARD_WR_DIFF.get((char_id, card_id), 0)
    threshold = SYS_THRESHOLD if card_id <= 20 else SKILL_THRESHOLD
    if freq >= threshold:
        base = freq
    else:
        base = max(FREQ_FLOOR, freq / FREQ_DIVISOR)
    if wr_diff < -5.0:
        base *= 0.3
    elif wr_diff < -2.0:
        base *= 0.6
    return base

AVG_SYS_PCT = {cid: char_sys_total[cid] / max(1, 20 * char_deck_count[cid]) for cid in range(20)}
AVG_SKILL_PCT = {cid: char_skill_total[cid] / max(1, 20 * char_deck_count[cid]) for cid in range(20)}
AVG_SPELL_PCT = {cid: char_spell_total[cid] / max(1, 20 * char_deck_count[cid]) for cid in range(20)}
AVG_UNIQUE = {cid: char_unique_total[cid] / max(1, char_deck_count[cid]) for cid in range(20)}

FORCED_QUANTITIES = {19: [0, 4]}

PENALTY = 0.03
PENALTY_CONC = 0.005


def parse_card_counts(cards_str):
    if pd.isna(cards_str):
        return {}
    return Counter(int(c) for c in cards_str.split('|'))


def cards_to_vector(char_id, card_counter, elo_diff=0):
    vec = np.zeros(N_FEATURES + 6, dtype=float)
    vec[0] = char_id
    total_system = 0
    total_skill = 0
    total_spell = 0
    for card_id, count in card_counter.items():
        key = (char_id, card_id) if card_id > 20 else ('system', card_id)
        if key in INTERNAL_CARD_INDEX:
            vec[INTERNAL_CARD_INDEX[key] + 1] = count ** 1.1
        if card_id <= 20:
            total_system += count
        elif card_id <= 114:
            total_skill += count
        else:
            total_spell += count
    vec[-5] = elo_diff
    vec[-4] = len(card_counter)
    vec[-3] = total_system
    vec[-2] = total_skill
    vec[-1] = total_spell
    return vec, total_system, total_skill, total_spell


# 2 - FEATURE ENGINEERING
print("Construindo matriz de features...")
X_rows = []
y_rows = []
weights = []

for _, row in df_filtered.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_cards = parse_card_counts(row['serverCards'])
    c_cards = parse_card_counts(row['clientCards'])
    elo_diff = row['serverUserElo'] - row['clientUserElo']

    s_vec, s_sys, s_skill, s_spell = cards_to_vector(s_char, s_cards, elo_diff=+elo_diff)
    c_vec, c_sys, c_skill, c_spell = cards_to_vector(c_char, c_cards, elo_diff=-elo_diff)

    X_rows.append(s_vec)
    y_rows.append(row['winner_binary'])
    w_s = 1.0 + 3.0 * (s_skill / 20) + 3.0 * (s_spell / 20)
    weights.append(w_s)

    X_rows.append(c_vec)
    y_rows.append(1 - row['winner_binary'])
    w_c = 1.0 + 3.0 * (c_skill / 20) + 3.0 * (c_spell / 20)
    weights.append(w_c)

X = np.array(X_rows)
y = np.array(y_rows)
weights = np.array(weights)
print(f"Exemplos de treinamento: {len(X)}")

# 3 - TREINAR MODELO
MODEL_PATH = Path(__file__).parent / "models" / "deck_model.pkl"

if MODEL_PATH.exists():
    print("Carregando modelo de deck do disco...")
    model = joblib.load(MODEL_PATH)
else:
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=w_train)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Modelo de deck salvo em models/deck_model.pkl")

y_pred = model.predict(X_test) if 'X_test' in dir() else model.predict(X[:100])
acc = accuracy_score(y_test[:100] if 'y_test' in dir() else y[:100], y_pred[:100])
print(f"Acuracia do modelo: {acc:.4f}")


def predict_match(server_elo, client_elo, server_char, client_char, server_cards, client_cards):
    elo_diff = server_elo - client_elo
    s_counter = Counter(server_cards) if not isinstance(server_cards, Counter) else server_cards
    c_counter = Counter(client_cards) if not isinstance(client_cards, Counter) else client_cards
    s_vec, _, _, _ = cards_to_vector(server_char, s_counter, elo_diff=+elo_diff)
    c_vec, _, _, _ = cards_to_vector(client_char, c_counter, elo_diff=-elo_diff)
    prob_s = model.predict_proba([s_vec])[0][1]
    prob_c = model.predict_proba([c_vec])[0][1]
    p_server = prob_s / (prob_s + prob_c) if (prob_s + prob_c) > 0 else 0.5
    return {
        'winner': 'server' if p_server >= 0.5 else 'client',
        'server_win_probability': p_server,
        'client_win_probability': 1 - p_server,
    }

CHAR_BEST_DECKS = {}

POP_SIZE = 100
GENERATIONS = 50
TOURNAMENT_K = 5
MUTATION_RATE = 0.10
ELITISM_COUNT = 10


def get_slot_skills(char_id):
    slots = {}
    for (cid, card_id), slot in SKILL_SLOT.items():
        if cid == char_id:
            slots.setdefault(slot, []).append(card_id)
    return slots


def _apply_forced(deck, char_id):
    available = sorted(CHAR_CARDS[char_id])
    weights = [card_weight(char_id, c) for c in available]
    for card_id, allowed in FORCED_QUANTITIES.items():
        if card_id not in CHAR_CARDS[char_id]:
            continue
        current = deck.count(card_id)
        if current in allowed:
            continue
        deck = [c for c in deck if c != card_id]
        target = random.choice(allowed)
        for _ in range(target):
            deck.append(card_id)
    while len(deck) < 20:
        deck.append(random.choices(available, weights=weights, k=1)[0])
    while len(deck) > 20:
        for i, c in enumerate(deck):
            if c not in FORCED_QUANTITIES:
                deck.pop(i)
                break
    return deck


def random_deck(char_id):
    slot_skills = get_slot_skills(char_id)
    available = sorted(CHAR_CARDS[char_id])
    sys_spell_ids = [c for c in available if c <= 20 or c >= 200]
    sys_spell_weights = [card_weight(char_id, c) for c in sys_spell_ids]

    def _pick_skill(candidates):
        w = [card_weight(char_id, s) for s in candidates]
        return random.choices(candidates, weights=w, k=1)[0]

    if CHAR_REAL_DECKS[char_id] and random.random() < 0.5:
        deck = random.choice(CHAR_REAL_DECKS[char_id])[:]
        deck = deck[:20]
        while len(deck) < 20:
            deck.append(random.choices(sys_spell_ids, weights=sys_spell_weights, k=1)[0])
        return _apply_forced(deck, char_id)

    deck = []
    counts = {}

    for skills in slot_skills.values():
        card = _pick_skill(skills)
        copies = random.randint(1, min(4, 20 - len(deck)))
        for _ in range(copies):
            deck.append(card)
        counts[card] = copies

    while len(deck) < 20:
        if not sys_spell_ids:
            break
        card = random.choices(sys_spell_ids, weights=sys_spell_weights, k=1)[0]
        if counts.get(card, 0) < 4:
            deck.append(card)
            counts[card] = counts.get(card, 0) + 1

    return _apply_forced(deck, char_id)


def repair_deck(deck, char_id):
    counts = Counter(deck)
    available = sorted(CHAR_CARDS[char_id])
    weights = [card_weight(char_id, c) for c in available]
    slot_skills = get_slot_skills(char_id)

    slot_cards = {}
    for card_id in list(deck):
        key = (char_id, card_id)
        if key in SKILL_SLOT:
            slot = SKILL_SLOT[key]
            if slot not in slot_cards:
                slot_cards[slot] = set()
            slot_cards[slot].add(card_id)

    for slot, cards in slot_cards.items():
        if len(cards) > 1:
            keep = random.choice(list(cards))
            for card_id in cards:
                if card_id != keep:
                    while card_id in deck:
                        deck.remove(card_id)
                        counts[card_id] -= 1

    for card_id in list(deck):
        if counts[card_id] > 4:
            deck.remove(card_id)
            counts[card_id] -= 1

    while len(deck) < 20:
        card = random.choices(available, weights=weights, k=1)[0]
        if counts.get(card, 0) < 4:
            deck.append(card)
            counts[card] = counts.get(card, 0) + 1

    return _apply_forced(deck, char_id)


def mutate(deck, char_id):
    available = sorted(CHAR_CARDS[char_id])
    slot_skills = get_slot_skills(char_id)

    for i in range(len(deck)):
        if random.random() < MUTATION_RATE / 20:
            old = deck[i]
            if old in FORCED_QUANTITIES:
                continue
            key = (char_id, old)
            if key in SKILL_SLOT:
                slot = SKILL_SLOT[key]
                candidates = [c for c in slot_skills[slot] if c != old and deck.count(c) < 4]
                if candidates:
                    w = [card_weight(char_id, c) for c in candidates]
                    deck[i] = random.choices(candidates, weights=w, k=1)[0]
                else:
                    candidates = [c for c in available if c != old and deck.count(c) < 4
                                  and ((char_id, c) not in SKILL_SLOT or deck.count(c) == 0)]
                    if candidates:
                        w = [card_weight(char_id, c) for c in candidates]
                        deck[i] = random.choices(candidates, weights=w, k=1)[0]
            else:
                candidates = [c for c in available if c != old and deck.count(c) < 4]
                if candidates:
                    w = [card_weight(char_id, c) for c in candidates]
                    deck[i] = random.choices(candidates, weights=w, k=1)[0]
    return deck


def decks_to_matrix(char_id, decks, elo_diffs=None):
    n = len(decks)
    mat = np.zeros((n, N_FEATURES + 6), dtype=float)
    mat[:, 0] = char_id
    if elo_diffs is not None:
        mat[:, -5] = elo_diffs
    for i, deck in enumerate(decks):
        total_system = 0
        total_skill = 0
        total_spell = 0
        card_counts = {}
        for card_id in deck:
            key = (char_id, card_id) if card_id > 20 else ('system', card_id)
            card_counts[key] = card_counts.get(key, 0) + 1
            if card_id <= 20:
                total_system += 1
            elif card_id <= 114:
                total_skill += 1
            else:
                total_spell += 1
        for key, count in card_counts.items():
            if key in INTERNAL_CARD_INDEX:
                mat[i, INTERNAL_CARD_INDEX[key] + 1] = count ** 1.1
        mat[i, -4] = len(card_counts)
        mat[i, -3] = total_system
        mat[i, -2] = total_skill
        mat[i, -1] = total_spell
    return mat


def evaluate_population(char_id, decks):
    mat = decks_to_matrix(char_id, decks)
    probs = model.predict_proba(mat)[:, 1]
    for i, deck in enumerate(decks):
        sys_cnt = sum(1 for c in deck if c <= 20)
        skill_cnt = sum(1 for c in deck if 100 <= c <= 114)
        spell_cnt = len(deck) - sys_cnt - skill_cnt
        deviation = (abs(sys_cnt / 20 - AVG_SYS_PCT[char_id]) +
                     abs(skill_cnt / 20 - AVG_SKILL_PCT[char_id]) +
                     abs(spell_cnt / 20 - AVG_SPELL_PCT[char_id]))
        n_unique = len(set(deck))
        conc_deviation = abs(n_unique - AVG_UNIQUE[char_id]) / 20
        probs[i] = max(0, min(1, probs[i] - PENALTY * deviation - PENALTY_CONC * conc_deviation))
    return probs


def tournament_select(decks, scores, k=5):
    idxs = random.sample(range(len(decks)), k)
    best_idx = max(idxs, key=lambda i: scores[i])
    return decks[best_idx][:]


def crossover(p1, p2):
    return [p1[i] if random.random() < 0.5 else p2[i] for i in range(20)]


if __name__ == "__main__":
    print("\nMedias reais de cartas unicas por personagem:")
    for cid in range(20):
        avg_u = AVG_UNIQUE.get(cid, 0)
        n_decks = char_deck_count.get(cid, 0)
        print(f"  {CHARACTERS[cid]}: {avg_u:.1f} unicas  ({n_decks} decks)")

    total_cards = sum(len(CHAR_CARDS[cid]) for cid in range(20))
    below_2 = sum(1 for cid in range(20) for card_id in CHAR_CARDS[cid]
                  if CHAR_CARD_FREQ.get((cid, card_id), 0) < SKILL_THRESHOLD)
    print(f"\nCartas com peso reduzido (freq < {SKILL_THRESHOLD}% skills, < {SYS_THRESHOLD}% system): {below_2}/{total_cards}")
    print(f"FORCED_QUANTITIES: {FORCED_QUANTITIES}")

    # 4 - ALGORITMO GENETICO

    print("\n" + "=" * 70)
    print("ALGORITMO GENETICO - OTIMIZACAO DE BARALHOS")
    print("=" * 70)

    for char_id in range(20):
        char_name = CHARACTERS[char_id]
        print(f"\n--- {char_name} (ID {char_id}) ---")

        population = [random_deck(char_id) for _ in range(POP_SIZE)]

        for gen in range(GENERATIONS):
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

            if gen % 10 == 0 or gen == GENERATIONS - 1:
                best = max(scores)
                print(f"  Gen {gen:3d}: best WR = {best:.4f}")

        final_scores = evaluate_population(char_id, population)
        ranked = sorted(zip(final_scores, population), key=lambda x: x[0], reverse=True)

        seen = set()
        unique_decks = []
        for prob, deck in ranked:
            key = tuple(sorted(Counter(deck).items()))
            if key not in seen:
                seen.add(key)
                unique_decks.append((prob, deck))
            if len(unique_decks) >= 3:
                break

        for rank, (prob, deck) in enumerate(unique_decks, 1):
            print(f"\n  Deck #{rank} (WR estimada: {prob:.1%}):")
            counts = Counter(deck)
            for card_id, count in sorted(counts.items()):
                name = get_card_name(char_id, card_id)
                print(f"    {count}x {name}")
            if rank == 1:
                CHAR_BEST_DECKS[char_id] = list(deck)
