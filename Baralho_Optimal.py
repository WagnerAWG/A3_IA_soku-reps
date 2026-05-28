import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from game_data import CHARACTERS, CHAR_CARDS, get_card_name
import random

random.seed(42)
np.random.seed(42)

# 1 - CARREGAR DADOS
print("Carregando dados...")
df = pd.read_csv('data.csv')

mask = (df['serverUserElo'] >= 1600) & (df['clientUserElo'] >= 1600)
df_filtered = df[mask].copy()
print(f"Partidas com ambos ELO >= 1600: {len(df_filtered)}")

df_filtered['winner_binary'] = (df_filtered['winner'] == -1).astype(int)

# 2 - FEATURE ENGINEERING
ALL_CARD_IDS = sorted(
    {c for cards_set in CHAR_CARDS.values() for c in cards_set}
)
CARD_INDEX = {card_id: idx for idx, card_id in enumerate(ALL_CARD_IDS)}
N_CARDS = len(ALL_CARD_IDS)

print(f"Total de cartas unicas no dataset: {N_CARDS}")


def parse_card_counts(cards_str):
    if pd.isna(cards_str):
        return {}
    return Counter(int(c) for c in cards_str.split('|'))


def cards_to_vector(char_id, card_counter):
    vec = np.zeros(N_CARDS + 1, dtype=int)
    vec[0] = char_id
    for card_id, count in card_counter.items():
        if card_id in CARD_INDEX:
            vec[CARD_INDEX[card_id] + 1] = count
    return vec


print("Construindo matriz de features...")
X_rows = []
y_rows = []

for _, row in df_filtered.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_cards = parse_card_counts(row['serverCards'])
    c_cards = parse_card_counts(row['clientCards'])

    X_rows.append(cards_to_vector(s_char, s_cards))
    y_rows.append(row['winner_binary'])

    X_rows.append(cards_to_vector(c_char, c_cards))
    y_rows.append(1 - row['winner_binary'])

X = np.array(X_rows)
y = np.array(y_rows)

print(f"Exemplos de treinamento: {len(X)}")

# 3 - TREINAR MODELO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acuracia do modelo: {acc:.4f}")

# 4 - ALGORITMO GENETICO

POP_SIZE = 100
GENERATIONS = 50
TOURNAMENT_K = 5
MUTATION_RATE = 0.10
ELITISM_COUNT = 10


def random_deck(char_id):
    available = sorted(CHAR_CARDS[char_id])
    deck = []
    counts = {}
    for _ in range(20):
        while True:
            card = random.choice(available)
            if counts.get(card, 0) < 4:
                counts[card] = counts.get(card, 0) + 1
                deck.append(card)
                break
    return deck


def repair_deck(deck, char_id):
    counts = Counter(deck)
    available = sorted(CHAR_CARDS[char_id])
    for card in list(deck):
        if counts[card] > 4:
            deck.remove(card)
            counts[card] -= 1
            while True:
                c = random.choice(available)
                if counts.get(c, 0) < 4:
                    deck.append(c)
                    counts[c] = counts.get(c, 0) + 1
                    break
    return deck


def mutate(deck, char_id):
    available = sorted(CHAR_CARDS[char_id])
    for i in range(20):
        if random.random() < MUTATION_RATE / 20:
            old = deck[i]
            candidates = [c for c in available if c != old and deck.count(c) < 4]
            if candidates:
                deck[i] = random.choice(candidates)
    return deck


def decks_to_matrix(char_id, decks):
    n = len(decks)
    mat = np.zeros((n, N_CARDS + 1), dtype=int)
    mat[:, 0] = char_id
    for i, deck in enumerate(decks):
        for c in deck:
            if c in CARD_INDEX:
                mat[i, CARD_INDEX[c] + 1] += 1
    return mat


def evaluate_population(char_id, decks):
    mat = decks_to_matrix(char_id, decks)
    probs = model.predict_proba(mat)[:, 1]
    return probs


def tournament_select(decks, scores, k=TOURNAMENT_K):
    idxs = random.sample(range(len(decks)), k)
    best_idx = max(idxs, key=lambda i: scores[i])
    return decks[best_idx][:]


def crossover(p1, p2):
    child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(20)]
    return child


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
