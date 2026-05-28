import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
from game_data import CHARACTERS

df = pd.read_csv('data.csv')


df['winner_binary'] = (df['winner'] == -1).astype(int)
y = df['winner_binary']


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


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)



print(model.score(x_test, y_test))
print(classification_report(y_test, y_pred))


# Prediction helper function
def predict_match_winner(server_rank, client_rank, server_char, client_char):
    """
    Predict the match winner based on ranks and characters.
    
    Args:
        server_rank: Server player ELO ranking (numeric)
        client_rank: Client player ELO ranking (numeric)
        server_char: Server character ID (0-19)
        client_char: Client character ID (0-19)
    
    Returns:
        dict with 'winner' (1 for server, 0 for client) and 'probability'
    """
    default_strength = sum(character_strength.values()) / len(character_strength)
    
    # Create feature array matching the training format
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


# ===== TEST CASES WITH SPECIFIC FEATURE VALUES =====
print("\n" + "="*60)
print("TEST CASES: AI PREDICTIONS WITH SPECIFIC FEATURE VALUES")
print("="*60)

test_cases = [
    {
        'name': 'Test 1: Server stronger (rank & character)',
        'server_rank': 1700,
        'client_rank': 1500,
        'server_char': 13,  # Strength 10 (best WR=78.29%)
        'client_char': 10,  # Strength 1 (worst WR=30.32%)
    },
    {
        'name': 'Test 2: Client stronger (rank & character)',
        'server_rank': 1400,
        'client_rank': 1800,
        'server_char': 10,  # Strength 1 (worst)
        'client_char': 13,  # Strength 10 (best)
    },
    {
        'name': 'Test 3: Same rank, server character stronger',
        'server_rank': 1600,
        'client_rank': 1600,
        'server_char': 5,   # Strength 7 (strong, WR=64.82%)
        'client_char': 2,   # Strength 2 (weak, WR=36.00%)
    },
    {
        'name': 'Test 4: Same rank, client character stronger',
        'server_rank': 1600,
        'client_rank': 1600,
        'server_char': 2,   # Strength 2 (weak)
        'client_char': 5,   # Strength 7 (strong)
    },
    {
        'name': 'Test 5: Server rank advantage vs client character advantage',
        'server_rank': 1750,
        'client_rank': 1550,
        'server_char': 7,   # Strength 5 (average, WR=51.50%)
        'client_char': 13,  # Strength 10 (best)
    },
    {
        'name': 'Test 6: Evenly matched (same rank, same strength characters)',
        'server_rank': 1650,
        'client_rank': 1650,
        'server_char': 1,   # Strength 5 (WR=52.89%)
        'client_char': 7,   # Strength 5 (WR=51.50%)
    },
    {
        'name': 'Test 7: Low rank vs high rank (same character)',
        'server_rank': 1200,
        'client_rank': 1900,
        'server_char': 5,   # Strength 7 (same character)
        'client_char': 5,   # Strength 7 (same character)
    },
    {
        'name': 'Test 8: High rank weak character vs low rank strong character',
        'server_rank': 1800,
        'client_rank': 1200,
        'server_char': 10,  # Strength 1 (worst)
        'client_char': 13,  # Strength 10 (best)
    },
    {
        'name': 'Test 9: Extreme rank diff + worst vs best character',
        'server_rank': 2800,
        'client_rank': 800,
        'server_char': 10,  # Strength 1 (worst)
        'client_char': 13,  # Strength 10 (best)
    },
    {
        'name': 'Test 10: Inverse extreme rank diff + best vs worst character',
        'server_rank': 800,
        'client_rank': 3200,
        'server_char': 13,  # Strength 10 (best)
        'client_char': 10,  # Strength 1 (worst)
    },
]

# Run all test cases
for test in test_cases:
    s_name = CHARACTERS.get(test['server_char'], f"Char {test['server_char']}")
    c_name = CHARACTERS.get(test['client_char'], f"Char {test['client_char']}")
    print(f"\n{test['name']}")
    print(f"  Server: Rank {test['server_rank']}, {s_name} (ID {test['server_char']})")
    print(f"  Client: Rank {test['client_rank']}, {c_name} (ID {test['client_char']})")
    
    result = predict_match_winner(
        test['server_rank'],
        test['client_rank'],
        test['server_char'],
        test['client_char']
    )
    
    print(f"  → PREDICTION: {result['winner'].upper()} wins")
    print(f"    - Server win probability: {result['server_win_probability']:.2%}")
    print(f"    - Client win probability: {result['client_win_probability']:.2%}")

print("\n" + "="*60)
