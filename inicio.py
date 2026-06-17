import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from game_data import CHARACTERS
from pathlib import Path
import joblib

# 1 - CARREGAR DADOS
df = pd.read_csv('data.csv')

df['winner_binary'] = (df['winner'] == -1).astype(int)
y = df['winner_binary']

# 2 - FEATURE ENGINEERING
from collections import defaultdict

mask = (df['serverUserElo'] >= 1500) & (df['clientUserElo'] >= 1500)
high_elo = df[mask].copy()

player_games = defaultdict(int)
player_wins = defaultdict(float)

for _, row in high_elo.iterrows():
    s_id = row['serverUserId']
    c_id = row['clientUserId']
    player_games[s_id] += 1
    player_games[c_id] += 1
    if row['winner'] == -1:
        player_wins[s_id] += 1
    else:
        player_wins[c_id] += 1

MIN_GAMES = 5
player_wr = {}
for pid, games in player_games.items():
    player_wr[pid] = player_wins[pid] / games if games >= MIN_GAMES else 0.5

char_contrib_sum = defaultdict(float)
char_contrib_weight = defaultdict(float)
char_players = defaultdict(set)

for _, row in high_elo.iterrows():
    s_id = row['serverUserId']
    c_id = row['clientUserId']
    s_char = int(row['serverCharacter'])
    c_char = int(row['clientCharacter'])
    s_won = row['winner'] == -1

    w_s = np.sqrt(player_games[s_id])
    char_contrib_sum[s_char] += ((1 if s_won else 0) - player_wr[s_id]) * w_s
    char_contrib_weight[s_char] += w_s
    char_players[s_char].add(s_id)

    w_c = np.sqrt(player_games[c_id])
    char_contrib_sum[c_char] += ((0 if s_won else 1) - player_wr[c_id]) * w_c
    char_contrib_weight[c_char] += w_c
    char_players[c_char].add(c_id)

STRENGTH_PATH = Path(__file__).parent / "models" / "character_strength.pkl"

if STRENGTH_PATH.exists():
    character_strength = joblib.load(STRENGTH_PATH)
    print("Character Strength carregada do disco.")
else:
    character_strength = {}
    for cid in range(20):
        if char_contrib_weight[cid] > 0:
            raw = 1.0 + char_contrib_sum[cid] / char_contrib_weight[cid]
            n_players = len(char_players[cid])
            if n_players < 5:
                raw = (raw * n_players + 1.0 * 3) / (n_players + 3)
            character_strength[cid] = raw
        else:
            character_strength[cid] = 1.0
    STRENGTH_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(character_strength, STRENGTH_PATH)
    print("Character Strength salva em models/character_strength.pkl")

print("Character Strength (player-adjusted, ELO >= 1500):")
print("  (>1.0 = jogadores ganham mais com esse char, <1.0 = ganham menos)")
for cid in sorted(character_strength.keys()):
    name = CHARACTERS.get(int(cid), f"Char {cid}")
    n_players = len(char_players.get(cid, set()))
    games = int(char_contrib_weight.get(cid, 0))
    print(f"  {name}: {character_strength[cid]:.4f}  ({n_players} jogadores, ~{games} partidas)")

df['serverCharStrength'] = df['serverCharacter'].map(character_strength)
df['clientCharStrength'] = df['clientCharacter'].map(character_strength)

df['elo_diff'] = df['serverUserElo'] - df['clientUserElo']
df['strength_diff'] = df['serverCharStrength'] - df['clientCharStrength']

cols = [
    'elo_diff',
    'serverCharacter',
    'clientCharacter',
    'strength_diff',
]

x = df[cols]

# Data augmentation: versao espelhada (server <-> client, winner invertido)
x_mirror = x.copy()
x_mirror['elo_diff'] = -x_mirror['elo_diff']
x_mirror['strength_diff'] = -x_mirror['strength_diff']
x_mirror[['serverCharacter', 'clientCharacter']] = x_mirror[['clientCharacter', 'serverCharacter']].values
y_mirror = 1 - y

x = pd.concat([x, x_mirror], ignore_index=True)
y = pd.concat([y, y_mirror], ignore_index=True)

print(f"Exemplos de treinamento (com augment): {len(x)} (original: {len(df)})")

# 3 - TREINAR MODELO
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)



print(model.score(x_test, y_test))
print(classification_report(y_test, y_pred))


# 4 - FUNCAO DE PREDICAO
def predict_match_winner(server_rank, client_rank, server_char, client_char):
    """
    Preve o vencedor da partida com base no ELO e personagens.

    Args:
        server_rank: ELO do jogador servidor
        client_rank: ELO do jogador cliente
        server_char: ID do personagem do servidor (0-19)
        client_char: ID do personagem do cliente (0-19)

    Returns:
        dict com 'winner' ('server' ou 'client') e probabilidades
    """
    default_strength = sum(character_strength.values()) / len(character_strength)

    server_char_strength = character_strength.get(server_char, default_strength)
    client_char_strength = character_strength.get(client_char, default_strength)

    elo_diff = server_rank - client_rank
    strength_diff = server_char_strength - client_char_strength

    features = [[elo_diff, server_char, client_char, strength_diff]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        'winner': 'server' if prediction == 1 else 'client',
        'server_win_probability': probability[1],
        'client_win_probability': probability[0]
    }


if __name__ == "__main__":
    # 5 - CASOS DE TESTE
    print("\n" + "="*60)
    print("CASOS DE TESTE: PREDICOES DA IA")
    print("="*60)

    test_cases = [
        {
            'name': 'Teste 1: Servidor mais forte (ELO e personagem)',
            'server_rank': 1700,
            'client_rank': 1500,
            'server_char': 13,
            'client_char': 10,
        },
        {
            'name': 'Teste 2: Cliente mais forte (ELO e personagem)',
            'server_rank': 1400,
            'client_rank': 1800,
            'server_char': 10,
            'client_char': 13,
        },
        {
            'name': 'Teste 3: Mesmo ELO, personagem do servidor mais forte',
            'server_rank': 1600,
            'client_rank': 1600,
            'server_char': 5,
            'client_char': 2,
        },
        {
            'name': 'Teste 4: Mesmo ELO, personagem do cliente mais forte',
            'server_rank': 1600,
            'client_rank': 1600,
            'server_char': 2,
            'client_char': 5,
        },
        {
            'name': 'Teste 5: Vantagem de ELO do servidor vs vantagem de personagem do cliente',
            'server_rank': 1750,
            'client_rank': 1550,
            'server_char': 7,
            'client_char': 13,
        },
        {
            'name': 'Teste 6: Equilibrado (mesmo ELO, personagens de forca similar)',
            'server_rank': 1650,
            'client_rank': 1650,
            'server_char': 1,
            'client_char': 7,
        },
        {
            'name': 'Teste 7: ELO baixo vs ELO alto (mesmo personagem)',
            'server_rank': 1200,
            'client_rank': 1900,
            'server_char': 5,
            'client_char': 5,
        },
        {
            'name': 'Teste 8: ELO alto + personagem fraco vs ELO baixo + personagem forte',
            'server_rank': 1800,
            'client_rank': 1200,
            'server_char': 10,
            'client_char': 13,
        },
        {
            'name': 'Teste 9: Diferenca extrema de ELO + pior vs melhor personagem',
            'server_rank': 2800,
            'client_rank': 800,
            'server_char': 10,
            'client_char': 13,
        },
        {
            'name': 'Teste 10: Diferenca extrema inversa + melhor vs pior personagem',
            'server_rank': 800,
            'client_rank': 3200,
            'server_char': 13,
            'client_char': 10,
        },
    ]

    for test in test_cases:
        s_name = CHARACTERS.get(test['server_char'], f"Char {test['server_char']}")
        c_name = CHARACTERS.get(test['client_char'], f"Char {test['client_char']}")
        print(f"\n{test['name']}")
        print(f"  Servidor: ELO {test['server_rank']}, {s_name} (ID {test['server_char']})")
        print(f"  Cliente: ELO {test['client_rank']}, {c_name} (ID {test['client_char']})")

        result = predict_match_winner(
            test['server_rank'],
            test['client_rank'],
            test['server_char'],
            test['client_char']
        )

        print(f"  -> PREDICAO: {result['winner'].upper()} vence")
        print(f"    - Prob. vitoria do servidor: {result['server_win_probability']:.2%}")
        print(f"    - Prob. vitoria do cliente: {result['client_win_probability']:.2%}")

    print("\n" + "="*60)

    # 6 - TESTES COM BARALHO (modelo do Baralho_Optimal)
    print("\n" + "="*60)
    print("TESTES DE IMPACTO DO BARALHO NA PREDICAO")
    print("="*60)

    from Baralho_Optimal import predict_match

    REIMU_OPT = [12,12,12, 101,101,101,101, 104,104,104,104, 110,110,110, 208,208, 209, 210,210,210]
    MARISA_OPT = [100,100,100,100, 101,101,101, 200,200, 202, 203,203, 205,205, 209,209,209, 211,211,211]
    SAKUYA_OPT = [0,0, 12,12,12, 101,101, 200,200,200,200, 201,201,201,201, 205,205, 208,208,208]
    IKU_OPT = [12,12,12, 104,104,104,104, 106,106,106, 105,105,105, 208,208,208,208, 211,211,211]
    REIMU_BAD = [0,0,0,0, 5,5,5,5, 3,3,3,3, 102,102,102,102, 103,103,103,103]
    MARISA_BAD = [1,1,1,1, 5,5,5,5, 104,104,104,104, 204,204,204,204, 110,110,110,110]
    IKU_BAD = [0,0,0,0, 5,5,5,5, 100,100,100,100, 200,200,200,200, 201,201,201,201]
    MEILING_BAD = [0,0,0,0, 5,5,5,5, 100,100,100,100, 202,202,202,202, 200,200,200,200]

    deck_tests = [
        {
            'name': 'Deck Teste 1: Mesmo ELO, mesmo char, deck otimo vs deck ruim',
            'server_rank': 1800, 'client_rank': 1800,
            'server_char': 0, 'client_char': 0,
            'server_cards': REIMU_OPT, 'client_cards': REIMU_BAD,
        },
        {
            'name': 'Deck Teste 2: ELO menor + deck otimo vs ELO maior + deck ruim',
            'server_rank': 1600, 'client_rank': 1700,
            'server_char': 13, 'client_char': 17,
            'server_cards': IKU_OPT, 'client_cards': MEILING_BAD,
        },
        {
            'name': 'Deck Teste 3: Comparacao direta (sem deck vs com deck)',
            'server_rank': 1650, 'client_rank': 1650,
            'server_char': 1, 'client_char': 2,
            'server_cards': MARISA_OPT, 'client_cards': SAKUYA_OPT,
        },
        {
            'name': 'Deck Teste 4: Ambos com deck otimo, ELO decide',
            'server_rank': 1900, 'client_rank': 1500,
            'server_char': 0, 'client_char': 0,
            'server_cards': REIMU_OPT, 'client_cards': REIMU_OPT,
        },
    ]

    for test in deck_tests:
        s_name = CHARACTERS.get(test['server_char'], f"Char {test['server_char']}")
        c_name = CHARACTERS.get(test['client_char'], f"Char {test['client_char']}")
        print(f"\n{test['name']}")
        print(f"  Servidor: {s_name} ELO {test['server_rank']} | {len(test['server_cards'])} cartas")
        print(f"  Cliente:  {c_name} ELO {test['client_rank']} | {len(test['client_cards'])} cartas")

        result_sem = predict_match_winner(
            test['server_rank'], test['client_rank'],
            test['server_char'], test['client_char']
        )
        print(f"  -> SEM deck: {result_sem['winner'].upper()} vence ({result_sem['server_win_probability']:.1%} vs {result_sem['client_win_probability']:.1%})")

        result_com = predict_match(
            test['server_rank'], test['client_rank'],
            test['server_char'], test['client_char'],
            test['server_cards'], test['client_cards']
        )
        print(f"  -> COM deck: {result_com['winner'].upper()} vence ({result_com['server_win_probability']:.1%} vs {result_com['client_win_probability']:.1%})")

    print("\n" + "="*60)
