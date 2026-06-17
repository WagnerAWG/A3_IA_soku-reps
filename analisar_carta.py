import pandas as pd
from collections import Counter

df = pd.read_csv('data.csv')

CARD_ID = 103
CHAR_ID = 2
ELO_MIN = 1600

mask = ((df['serverUserElo'] >= ELO_MIN) & (df['clientUserElo'] >= ELO_MIN) &
        ((df['serverCharacter'] == CHAR_ID) | (df['clientCharacter'] == CHAR_ID)))
df_f = df[mask].copy()

with_card_wins = 0
with_card_total = 0
without_card_wins = 0
without_card_total = 0
with_card_elos = []
without_card_elos = []
copy_dist = Counter()

for _, row in df_f.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])

    if s_char == CHAR_ID:
        cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
        won = row['winner'] == -1
        elo = row['serverUserElo']
    else:
        cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]
        won = row['winner'] == 1
        elo = row['clientUserElo']

    count = cards.count(CARD_ID)
    if count > 0:
        with_card_total += 1
        copy_dist[count] += 1
        if won:
            with_card_wins += 1
        with_card_elos.append(elo)
    else:
        without_card_total += 1
        if won:
            without_card_wins += 1
        without_card_elos.append(elo)

total = with_card_total + without_card_total
freq_pct = 100 * with_card_total / total

print(f"Sakuya - Vanishing Everything (ID {CARD_ID})")
print(f"ELO >= {ELO_MIN}")
print(f"Partidas analisadas: {total}")
print()
print(f"Frequencia da carta: {freq_pct:.1f}% ({with_card_total}/{total})")
print(f"  Distribuicao de copias:")
for copies in sorted(copy_dist):
    print(f"    {copies}x: {copy_dist[copies]} ({100*copy_dist[copies]/with_card_total:.1f}%)")
print()

wr_with = 100 * with_card_wins / with_card_total if with_card_total > 0 else 0
wr_without = 100 * without_card_wins / without_card_total if without_card_total > 0 else 0
print(f"Win rate COM a carta:     {wr_with:.1f}% ({with_card_wins}/{with_card_total})")
print(f"Win rate SEM a carta:     {wr_without:.1f}% ({without_card_wins}/{without_card_total})")
print(f"Diferenca:                {wr_with - wr_without:+.1f}%")
print()

if with_card_elos:
    avg_elo_with = sum(with_card_elos) / len(with_card_elos)
else:
    avg_elo_with = 0
if without_card_elos:
    avg_elo_without = sum(without_card_elos) / len(without_card_elos)
else:
    avg_elo_without = 0

print(f"ELO medio COM a carta:    {avg_elo_with:.0f}")
print(f"ELO medio SEM a carta:    {avg_elo_without:.0f}")
print(f"Diferenca de ELO:         {avg_elo_with - avg_elo_without:+.0f}")

players_with = set()
players_without = set()
for _, row in df_f.iterrows():
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    if s_char == CHAR_ID:
        cards = [int(c) for c in str(row['serverCards']).split('|') if c.isdigit()]
        uid = row['serverUserId']
    else:
        cards = [int(c) for c in str(row['clientCards']).split('|') if c.isdigit()]
        uid = row['clientUserId']
    if CARD_ID in cards:
        players_with.add(uid)
    else:
        players_without.add(uid)

print()
print(f"Jogadores unicos que usam:    {len(players_with)}")
print(f"Jogadores unicos que NAO usam: {len(players_without)}")
