import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from game_data import (
    CHARACTERS, CHAR_CARDS, get_card_name,
    INTERNAL_CARD_INDEX, N_FEATURES, SKILL_SLOT,
)
import random

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

for _, row in df_filtered.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
    c_cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]

    for char_id, cards in [(s_char, s_cards), (c_char, c_cards)]:
        CHAR_REAL_DECKS[char_id].append(cards)
        char_sys_total[char_id] += sum(1 for c in cards if c <= 20)
        char_skill_total[char_id] += sum(1 for c in cards if 100 <= c <= 114)
        char_spell_total[char_id] += sum(1 for c in cards if c >= 200)
        char_deck_count[char_id] += 1

AVG_SYS_PCT = {cid: char_sys_total[cid] / max(1, 20 * char_deck_count[cid]) for cid in range(20)}
AVG_SKILL_PCT = {cid: char_skill_total[cid] / max(1, 20 * char_deck_count[cid]) for cid in range(20)}
AVG_SPELL_PCT = {cid: char_spell_total[cid] / max(1, 20 * char_deck_count[cid]) for cid in range(20)}

PENALTY = 0.03


def parse_card_counts(cards_str):
    if pd.isna(cards_str):
        return {}
    return Counter(int(c) for c in cards_str.split('|'))


def cards_to_vector(char_id, card_counter):
    vec = np.zeros(N_FEATURES + 4, dtype=int)
    vec[0] = char_id
    total_system = 0
    total_skill = 0
    total_spell = 0
    for card_id, count in card_counter.items():
        key = (char_id, card_id) if card_id > 20 else ('system', card_id)
        if key in INTERNAL_CARD_INDEX:
            vec[INTERNAL_CARD_INDEX[key] + 1] = count
        if card_id <= 20:
            total_system += count
        elif card_id <= 114:
            total_skill += count
        else:
            total_spell += count
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

    s_vec, s_sys, s_skill, s_spell = cards_to_vector(s_char, s_cards)
    c_vec, c_sys, c_skill, c_spell = cards_to_vector(c_char, c_cards)

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
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, sample_weight=w_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acuracia do modelo: {acc:.4f}")

# 4 - ALGORITMO GENETICO

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


def random_deck(char_id):
    if CHAR_REAL_DECKS[char_id] and random.random() < 0.5:
        deck = random.choice(CHAR_REAL_DECKS[char_id])[:]
        available = sorted(CHAR_CARDS[char_id])
        while len(deck) < 20:
            deck.append(random.choice(available))
        return deck[:20]

    available = sorted(CHAR_CARDS[char_id])
    slot_skills = get_slot_skills(char_id)
    system_spell_ids = [c for c in available if c <= 20 or c >= 200]

    deck = []
    counts = {}

    for skills in slot_skills.values():
        card = random.choice(skills)
        copies = random.randint(1, min(4, 20 - len(deck)))
        for _ in range(copies):
            deck.append(card)
        counts[card] = copies

    while len(deck) < 20:
        card = random.choice(system_spell_ids)
        if counts.get(card, 0) < 4:
            deck.append(card)
            counts[card] = counts.get(card, 0) + 1

    return deck


def repair_deck(deck, char_id):
    counts = Counter(deck)
    available = sorted(CHAR_CARDS[char_id])
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
        card = random.choice(available)
        if counts.get(card, 0) < 4:
            deck.append(card)
            counts[card] = counts.get(card, 0) + 1

    return deck


def mutate(deck, char_id):
    available = sorted(CHAR_CARDS[char_id])
    slot_skills = get_slot_skills(char_id)

    for i in range(len(deck)):
        if random.random() < MUTATION_RATE / 20:
            old = deck[i]
            key = (char_id, old)
            if key in SKILL_SLOT:
                slot = SKILL_SLOT[key]
                candidates = [c for c in slot_skills[slot] if c != old and deck.count(c) < 4]
                if not candidates:
                    candidates = [c for c in available if c != old and deck.count(c) < 4
                                  and ((char_id, c) not in SKILL_SLOT or deck.count(c) == 0)]
            else:
                candidates = [c for c in available if c != old and deck.count(c) < 4]

            if candidates:
                deck[i] = random.choice(candidates)
    return deck


def decks_to_matrix(char_id, decks):
    n = len(decks)
    mat = np.zeros((n, N_FEATURES + 4), dtype=int)
    mat[:, 0] = char_id
    for i, deck in enumerate(decks):
        total_system = 0
        total_skill = 0
        total_spell = 0
        for card_id in deck:
            key = (char_id, card_id) if card_id > 20 else ('system', card_id)
            if key in INTERNAL_CARD_INDEX:
                mat[i, INTERNAL_CARD_INDEX[key] + 1] += 1
            if card_id <= 20:
                total_system += 1
            elif card_id <= 114:
                total_skill += 1
            else:
                total_spell += 1
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
        probs[i] = max(0, min(1, probs[i] - PENALTY * deviation))
    return probs


def tournament_select(decks, scores, k=TOURNAMENT_K):
    idxs = random.sample(range(len(decks)), k)
    best_idx = max(idxs, key=lambda i: scores[i])
    return decks[best_idx][:]


def crossover(p1, p2):
    return [p1[i] if random.random() < 0.5 else p2[i] for i in range(20)]


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
