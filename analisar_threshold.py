from collections import defaultdict
import pandas as pd

df = pd.read_csv('data.csv')

THRESHOLDS = [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

results = {t: {} for t in THRESHOLDS}

for t in THRESHOLDS:
    mask = (df['serverUserElo'] >= t) & (df['clientUserElo'] >= t)
    df_f = df[mask]

    char_unique_total = defaultdict(int)
    char_deck_count = defaultdict(int)

    for _, row in df_f.iterrows():
        s_char = int(row['serverCharacter'])
        c_char = int(row['clientCharacter'])
        s_cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
        c_cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]

        for char_id, cards in [(s_char, s_cards), (c_char, c_cards)]:
            char_unique_total[char_id] += len(set(cards))
            char_deck_count[char_id] += 1

    for cid in range(20):
        results[t][cid] = char_unique_total[cid] / max(1, char_deck_count[cid])

from game_data import CHARACTERS

print("AVG_UNIQUE por personagem x threshold de ELO")
print("=" * 120)
header = f"{'Personagem':<25} " + "".join(f"{'ELO>=' + str(t):>8}" for t in THRESHOLDS) + f"  {'Total decks (>={THRESHOLDS[0]})':>10}"
print(header)
print("-" * 120)

total_decks_1300 = {cid: 0 for cid in range(20)}
mask_1300 = (df['serverUserElo'] >= 1300) & (df['clientUserElo'] >= 1300)
df_1300 = df[mask_1300]
for _, row in df_1300.iterrows():
    total_decks_1300[int(row['serverCharacter'])] += 1
    total_decks_1300[int(row['clientCharacter'])] += 1

total_1300 = sum(total_decks_1300[cid] for cid in range(20))
for cid in range(20):
    name = CHARACTERS.get(cid, f"Char {cid}")
    row_str = f"{name:<25} "
    for t in THRESHOLDS:
        val = results[t].get(cid, 0)
        row_str += f"{val:>8.1f}"
    row_str += f"  {total_decks_1300[cid]:>10}"
    print(row_str)
    if cid == 2:
        print("-" * 120)

print("-" * 120)
print()

for cid in [2]:
    name = CHARACTERS.get(cid, f"Char {cid}")
    print(f"--- {name} ---")
    for t in THRESHOLDS:
        mask_t = (df['serverUserElo'] >= t) & (df['clientUserElo'] >= t)
        df_t = df[mask_t]
        matches = len(df_t[
            (df_t['serverCharacter'] == cid) | (df_t['clientCharacter'] == cid)
        ])
        decks_per = total_decks_1300[cid]
        if t == 1300:
            decks_t = decks_per
        else:
            decks_t = 0
            for _, row in df_t.iterrows():
                if int(row['serverCharacter']) == cid:
                    decks_t += 1
                if int(row['clientCharacter']) == cid:
                    decks_t += 1
        pct = 100 * decks_t / max(1, decks_per)
        print(f"  ELO >= {t}: {results[t][cid]:.1f} unicas  ({decks_t} decks, {pct:.1f}% do total)")
