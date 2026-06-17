from inicio import predict_match_winner, character_strength
from Baralho_Optimal import predict_match
from game_data import CHARACTERS

print("=" * 60)
print("TESTES UNIFICADOS")
print("=" * 60)

# ------ Bloco 1: ELO + personagem (sem deck) ------
print("\n" + "=" * 60)
print("BLOCO 1: PREDICAO POR ELO + PERSONAGEM")
print("=" * 60)

char_tests = [
    ('Teste 1: Servidor mais forte (ELO e personagem)', 1700, 13, 1500, 10),
    ('Teste 2: Cliente mais forte (ELO e personagem)', 1400, 10, 1800, 13),
    ('Teste 3: Mesmo ELO, personagem do servidor mais forte', 1600, 5, 1600, 2),
    ('Teste 4: Mesmo ELO, personagem do cliente mais forte', 1600, 2, 1600, 5),
    ('Teste 5: Vantagem de ELO do servidor vs vantagem de personagem do cliente', 1750, 7, 1550, 13),
    ('Teste 6: Equilibrado (mesmo ELO, personagens de forca similar)', 1650, 1, 1650, 7),
    ('Teste 7: ELO baixo vs ELO alto (mesmo personagem)', 1200, 5, 1900, 5),
    ('Teste 8: ELO alto + personagem fraco vs ELO baixo + personagem forte', 1800, 10, 1200, 13),
    ('Teste 9: Diferenca extrema de ELO + pior vs melhor personagem', 2800, 10, 800, 13),
    ('Teste 10: Diferenca extrema inversa + melhor vs pior personagem', 800, 13, 3200, 10),
]

for name, s_elo, s_char, c_elo, c_char in char_tests:
    s_name = CHARACTERS.get(s_char, f"Char {s_char}")
    c_name = CHARACTERS.get(c_char, f"Char {c_char}")
    print(f"\n{name}")
    print(f"  Servidor: ELO {s_elo}, {s_name} (ID {s_char})")
    print(f"  Cliente:  ELO {c_elo}, {c_name} (ID {c_char})")

    r = predict_match_winner(s_elo, c_elo, s_char, c_char)
    print(f"  -> {r['winner'].upper()} vence ({r['server_win_probability']:.1%} vs {r['client_win_probability']:.1%})")

# ------ Bloco 2: ELO + personagem + deck ------
print("\n" + "=" * 60)
print("BLOCO 2: IMPACTO DO BARALHO NA PREDICAO")
print("=" * 60)

REIMU_OPT = [12,12,12, 101,101,101,101, 104,104,104,104, 110,110,110, 208,208, 209, 210,210,210]
MARISA_OPT = [100,100,100,100, 101,101,101, 200,200, 202, 203,203, 205,205, 209,209,209, 211,211,211]
SAKUYA_OPT = [0,0, 12,12,12, 101,101, 200,200,200,200, 201,201,201,201, 205,205, 208,208,208]
IKU_OPT = [12,12,12, 104,104,104,104, 106,106,106, 105,105,105, 208,208,208,208, 211,211,211]
REIMU_BAD = [0,0,0,0, 5,5,5,5, 3,3,3,3, 102,102,102,102, 103,103,103,103]
MARISA_BAD = [1,1,1,1, 5,5,5,5, 104,104,104,104, 204,204,204,204, 110,110,110,110]
IKU_BAD = [0,0,0,0, 5,5,5,5, 100,100,100,100, 200,200,200,200, 201,201,201,201]
MEILING_BAD = [0,0,0,0, 5,5,5,5, 100,100,100,100, 202,202,202,202, 200,200,200,200]

deck_tests = [
    ('Deck Teste 1: Mesmo ELO, mesmo char, deck otimo vs deck ruim', 1800, 0, REIMU_OPT, 1800, 0, REIMU_BAD),
    ('Deck Teste 2: ELO menor + deck otimo vs ELO maior + deck ruim', 1600, 13, IKU_OPT, 1700, 17, MEILING_BAD),
    ('Deck Teste 3: Ambos deck otimo, Marisa vs Sakuya (mesmo ELO)', 1650, 1, MARISA_OPT, 1650, 2, SAKUYA_OPT),
    ('Deck Teste 4: Ambos deck otimo, ELO decide (Reimu 1900 vs 1500)', 1900, 0, REIMU_OPT, 1500, 0, REIMU_OPT),
]

for name, s_elo, s_char, s_cards, c_elo, c_char, c_cards in deck_tests:
    s_name = CHARACTERS.get(s_char, f"Char {s_char}")
    c_name = CHARACTERS.get(c_char, f"Char {c_char}")
    print(f"\n{name}")
    print(f"  Servidor: {s_name} ELO {s_elo} | {len(s_cards)} cartas")
    print(f"  Cliente:  {c_name} ELO {c_elo} | {len(c_cards)} cartas")

    r_sem = predict_match_winner(s_elo, c_elo, s_char, c_char)
    print(f"  -> SEM deck: {r_sem['winner'].upper()} vence ({r_sem['server_win_probability']:.1%} vs {r_sem['client_win_probability']:.1%})")

    r_com = predict_match(s_elo, c_elo, s_char, c_char, s_cards, c_cards)
    print(f"  -> COM deck: {r_com['winner'].upper()} vence ({r_com['server_win_probability']:.1%} vs {r_com['client_win_probability']:.1%})")

print("\n" + "=" * 60)
