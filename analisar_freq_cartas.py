import pandas as pd
from collections import defaultdict
from game_data import CHARACTERS, CHAR_CARDS, get_card_name, SYSTEM_CARDS

df = pd.read_csv('data.csv')
ELO_MIN = 1600

mask = (df['serverUserElo'] >= ELO_MIN) & (df['clientUserElo'] >= ELO_MIN)
df_f = df[mask].copy()

char_card_stats = {cid: {} for cid in range(20)}

for _, row in df_f.iterrows():
    chars_and_cards = []
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
    c_cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]

    s_won = row['winner'] == -1
    chars_and_cards.append((s_char, s_cards, s_won, row['serverUserElo']))
    chars_and_cards.append((c_char, c_cards, not s_won, row['clientUserElo']))

    for char_id, cards, won, elo in chars_and_cards:
        unique = set(cards)
        for card_id in unique:
            if card_id not in char_card_stats[char_id]:
                char_card_stats[char_id][card_id] = {
                    'total': 0, 'wins': 0, 'elo_sum': 0,
                }
            char_card_stats[char_id][card_id]['total'] += 1
            if won:
                char_card_stats[char_id][card_id]['wins'] += 1
            char_card_stats[char_id][card_id]['elo_sum'] += elo

char_deck_counts = {cid: 0 for cid in range(20)}
char_no_card_stats = {cid: {'total': 0, 'wins': 0, 'elo_sum': 0} for cid in range(20)}

for _, row in df_f.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
    c_cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]
    s_won = row['winner'] == -1
    c_won = not s_won

    s_set = set(s_cards)
    c_set = set(c_cards)

    for char_id, cards_set, won, elo in [(s_char, s_set, s_won, row['serverUserElo']),
                                          (c_char, c_set, c_won, row['clientUserElo'])]:
        char_deck_counts[char_id] += 1
        char_no_card_stats[char_id]['total'] += 1
        if won:
            char_no_card_stats[char_id]['wins'] += 1
        char_no_card_stats[char_id]['elo_sum'] += elo

BUCKETS = [(0, 2, "<2%"), (2, 5, "2-5%"), (5, 10, "5-10%"),
           (10, 15, "10-15%"), (15, 20, "15-20%"),
           (20, 30, "20-30%"), (30, 50, "30-50%"), (50, 100, ">50%")]

all_cards = []

for char_id in range(20):
    n_decks = char_deck_counts.get(char_id, 1)
    for card_id in sorted(CHAR_CARDS[char_id]):
        stats = char_card_stats[char_id].get(card_id)
        if stats is None:
            freq = 0
            wr_with = 0
            elo_with = 0
            count = 0
        else:
            count = stats['total']
            freq = 100 * count / n_decks
            wr_with = 100 * stats['wins'] / count if count > 0 else 0
            elo_with = stats['elo_sum'] / count if count > 0 else 0

        no_card = char_no_card_stats[char_id]
        decks_without = no_card['total'] - count
        if decks_without > 0:
            wins_without = no_card['wins'] - (stats['wins'] if stats else 0)
            elo_without = (no_card['elo_sum'] - (stats['elo_sum'] if stats else 0)) / decks_without
            wr_without = 100 * wins_without / decks_without
        else:
            wr_without = 0
            elo_without = 0

        wr_diff = wr_with - wr_without
        elo_diff = elo_with - elo_without

        name = get_card_name(char_id, card_id)

        bucket = None
        for lo, hi, label in BUCKETS:
            if lo <= freq < hi:
                bucket = label
                break
        if bucket is None:
            bucket = ">50%"

        all_cards.append({
            'char_id': char_id,
            'char_name': CHARACTERS[char_id],
            'card_id': card_id,
            'card_name': name,
            'freq': freq,
            'count': count,
            'n_decks': n_decks,
            'wr_with': wr_with,
            'wr_without': wr_without,
            'wr_diff': wr_diff,
            'elo_with': elo_with,
            'elo_without': elo_without,
            'elo_diff': elo_diff,
            'bucket': bucket,
        })

print("=" * 110)
print(f"ANALISE DE FREQUENCIA DE CARTAS (ELO >= {ELO_MIN})")
print("=" * 110)

bucket_stats = {label: {'count': 0, 'wr_diff_sum': 0, 'elo_diff_sum': 0, 'cards': []} for _, _, label in BUCKETS}

for card in all_cards:
    b = card['bucket']
    bucket_stats[b]['count'] += 1
    bucket_stats[b]['wr_diff_sum'] += card['wr_diff']
    bucket_stats[b]['elo_diff_sum'] += card['elo_diff']

print(f"\n{'Bucket':<12} {'Cartas':>6} {'WR diff media':>13} {'ELO diff medio':>14}")
print("-" * 50)
for lo, hi, label in BUCKETS:
    s = bucket_stats[label]
    n = s['count']
    avg_wr = s['wr_diff_sum'] / n if n > 0 else 0
    avg_elo = s['elo_diff_sum'] / n if n > 0 else 0
    print(f"{label:<12} {n:>6} {avg_wr:>+12.2f}% {avg_elo:>+13.0f}")

print()

SUSPECT_BUCKETS = ["<2%", "2-5%", "5-10%"]
print("--- Cartas com frequencia < 10% (suspeitas) ---")
print(f"{'Personagem':<25} {'Carta':<50} {'Freq':>6} {'WR diff':>8} {'ELO diff':>9}")
print("-" * 105)

suspect_cards = [c for c in all_cards if c['bucket'] in SUSPECT_BUCKETS]
suspect_cards.sort(key=lambda x: x['freq'])

for card in suspect_cards:
    print(f"{card['char_name']:<25} {card['card_name']:<50} {card['freq']:>5.1f}% {card['wr_diff']:>+7.2f}% {card['elo_diff']:>+8.0f}")

print()
print(f"Total de cartas suspeitas (<10%): {len(suspect_cards)} de {len(all_cards)} total")

print()

for cid in [2, 7, 13]:
    name = CHARACTERS[cid]
    n_decks = char_deck_counts[cid]
    print(f"--- {name} (ID {cid}): {n_decks} decks, {len([c for c in CHAR_CARDS[cid]])} cartas ---")
    char_cards = [c for c in all_cards if c['char_id'] == cid]
    char_cards.sort(key=lambda x: x['freq'])

    for card in char_cards:
        if card['freq'] <= 100:
            marker = ""
            if card['freq'] < 10:
                marker = "  <<< BAIXA"
            elif card['freq'] > 80:
                marker = "  *** ALTA"
            print(f"  {card['freq']:>5.1f}% {card['wr_diff']:>+6.2f}% WR  {card['card_name']}{marker}")
    print()
