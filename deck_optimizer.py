import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from game_data import CHAR_CARDS, get_card_name

random.seed(42)
np.random.seed(42)

DATA_FILE = Path(__file__).parent / "data.csv"


def parse_card_counts(cards_str):
    if pd.isna(cards_str):
        return {}
    return Counter(int(c) for c in str(cards_str).split("|") if c.strip())


class DeckOptimizer:
    def __init__(self):
        self.model = None
        self.all_card_ids = sorted({c for cards_set in CHAR_CARDS.values() for c in cards_set})
        self.card_index = {card_id: idx for idx, card_id in enumerate(self.all_card_ids)}
        self.n_cards = len(self.all_card_ids)
        self._cache = {}
        self._train_model()

    def _cards_to_vector(self, char_id, card_counter):
        vec = np.zeros(self.n_cards + 1, dtype=int)
        vec[0] = char_id
        for card_id, count in card_counter.items():
            if card_id in self.card_index:
                vec[self.card_index[card_id] + 1] = count
        return vec

    def _train_model(self):
        if not DATA_FILE.exists():
            return

        try:
            df = pd.read_csv(DATA_FILE)
            if 'winner' not in df.columns or 'serverCharacter' not in df.columns or 'clientCharacter' not in df.columns:
                return

            mask = (df['serverUserElo'] >= 1600) & (df['clientUserElo'] >= 1600)
            df_filtered = df[mask].copy()
            if df_filtered.empty:
                return

            df_filtered['winner_binary'] = (df_filtered['winner'] == -1).astype(int)

            X_rows = []
            y_rows = []

            for _, row in df_filtered.iterrows():
                server_count = parse_card_counts(row.get('serverCards', ''))
                client_count = parse_card_counts(row.get('clientCards', ''))

                X_rows.append(self._cards_to_vector(int(row['serverCharacter']), server_count))
                y_rows.append(row['winner_binary'])

                X_rows.append(self._cards_to_vector(int(row['clientCharacter']), client_count))
                y_rows.append(1 - row['winner_binary'])

            if not X_rows:
                return

            X = np.array(X_rows)
            y = np.array(y_rows)

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            self.model = model
        except Exception:
            self.model = None

    def random_deck(self, char_id):
        available = sorted(CHAR_CARDS[char_id])
        deck = []
        counts = {}
        while len(deck) < 20:
            card = random.choice(available)
            if counts.get(card, 0) < 4:
                deck.append(card)
                counts[card] = counts.get(card, 0) + 1
        return deck

    def repair_deck(self, deck, char_id):
        counts = Counter(deck)
        available = sorted(CHAR_CARDS[char_id])
        for card in list(deck):
            while counts[card] > 4:
                deck.remove(card)
                counts[card] -= 1
                replacement = random.choice(available)
                while counts.get(replacement, 0) >= 4:
                    replacement = random.choice(available)
                deck.append(replacement)
                counts[replacement] = counts.get(replacement, 0) + 1
        while len(deck) < 20:
            replacement = random.choice(available)
            if counts.get(replacement, 0) < 4:
                deck.append(replacement)
                counts[replacement] = counts.get(replacement, 0) + 1
        return deck

    def mutate(self, deck, char_id):
        available = sorted(CHAR_CARDS[char_id])
        counts = Counter(deck)
        for i in range(len(deck)):
            if random.random() < 0.05:
                old = deck[i]
                candidates = [c for c in available if c != old and counts.get(c, 0) < 4]
                if candidates:
                    new_card = random.choice(candidates)
                    counts[old] -= 1
                    deck[i] = new_card
                    counts[new_card] = counts.get(new_card, 0) + 1
        return deck

    def decks_to_matrix(self, char_id, decks):
        n = len(decks)
        mat = np.zeros((n, self.n_cards + 1), dtype=int)
        mat[:, 0] = char_id
        for i, deck in enumerate(decks):
            for c in deck:
                if c in self.card_index:
                    mat[i, self.card_index[c] + 1] += 1
        return mat

    def evaluate_population(self, char_id, decks):
        if self.model is None:
            return np.zeros(len(decks))
        mat = self.decks_to_matrix(char_id, decks)
        return self.model.predict_proba(mat)[:, 1]

    def tournament_select(self, decks, scores, k=5):
        idxs = random.sample(range(len(decks)), k)
        best_idx = max(idxs, key=lambda i: scores[i])
        return decks[best_idx][:]

    def crossover(self, p1, p2):
        return [p1[i] if random.random() < 0.5 else p2[i] for i in range(20)]

    def optimize_deck(self, char_id, pop_size=40, generations=20, elitism_count=5):
        if char_id in self._cache:
            return self._cache[char_id]
        if self.model is None:
            return None, None

        population = [self.random_deck(char_id) for _ in range(pop_size)]
        for _ in range(generations):
            scores = self.evaluate_population(char_id, population)
            ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
            new_pop = [deck for _, deck in ranked[:elitism_count]]
            while len(new_pop) < pop_size:
                if random.random() < 0.7:
                    p1 = self.tournament_select(population, scores)
                    p2 = self.tournament_select(population, scores)
                    child = self.crossover(p1, p2)
                else:
                    child = self.tournament_select(population, scores)
                child = self.repair_deck(self.mutate(child, char_id), char_id)
                new_pop.append(child)
            population = new_pop

        final_scores = self.evaluate_population(char_id, population)
        best_idx = int(np.argmax(final_scores))
        best_deck = population[best_idx]
        best_prob = float(final_scores[best_idx])
        self._cache[char_id] = (best_deck, best_prob)
        return best_deck, best_prob

    def format_deck(self, char_id, deck):
        counts = Counter(deck)
        return [
            {
                "card_id": card_id,
                "name": get_card_name(char_id, card_id),
                "count": count,
            }
            for card_id, count in sorted(counts.items())
        ]
