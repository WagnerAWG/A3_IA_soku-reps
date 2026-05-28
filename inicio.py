import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
from game_data import CHARACTERS

# 1 - CARREGAR DADOS
df = pd.read_csv('data.csv')

df['winner_binary'] = (df['winner'] == -1).astype(int)
y = df['winner_binary']

# 2 - FEATURE ENGINEERING
mask = (df['serverUserElo'] >= 1600) & (df['clientUserElo'] >= 1600)
high_elo = df[mask]

server_wins = high_elo[high_elo['winner'] == -1].groupby('serverCharacter').size()
server_total = high_elo.groupby('serverCharacter').size()
client_wins = high_elo[high_elo['winner'] == 1].groupby('clientCharacter').size()
client_total = high_elo.groupby('clientCharacter').size()

total_wins = server_wins.add(client_wins, fill_value=0)
total_matches = server_total.add(client_total, fill_value=0)
character_strength = (total_wins / total_matches).to_dict()

print("Character Strength (win rate with ELO >= 1600):")
for cid in sorted(character_strength.keys()):
    name = CHARACTERS.get(int(cid), f"Char {cid}")
    print(f"  {name}: {character_strength[cid]:.2%}")

df['serverCharStrength'] = df['serverCharacter'].map(character_strength)
df['clientCharStrength'] = df['clientCharacter'].map(character_strength)

cols = [
    'serverUserElo',
    'clientUserElo',
    'serverCharacter',
    'clientCharacter',
    'serverCharStrength',
    'clientCharStrength',
]

x = df[cols]

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
    
    features = [[server_rank, client_rank, server_char, client_char, server_char_strength, client_char_strength]]
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'winner': 'server' if prediction == 1 else 'client',
        'server_win_probability': probability[1],
        'client_win_probability': probability[0]
    }


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
