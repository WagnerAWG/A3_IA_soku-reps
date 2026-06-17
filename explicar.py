"""
Explica as decisoes do modelo:
  python explicar.py --char 0              Forca da personagem
  python explicar.py --card 2 103          Forca de uma carta (Sakuya, Vanishing Everything)
  python explicar.py --match 0 1 1800 1600 Reimu vs Marisa, ELO 1800 vs 1600
"""
import sys
import random
from collections import Counter
from game_data import CHARACTERS, CHAR_CARDS, get_card_name

random.seed(42)

# ------ Funcoes auxiliares ------

def _load_card_data(char_id, card_id):
    import pandas as pd
    from collections import defaultdict
    df = pd.read_csv('data.csv')
    mask = (df['serverUserElo'] >= 1600) & (df['clientUserElo'] >= 1600)
    df_f = df[mask]
    cc = defaultdict(int)
    cw = defaultdict(int)
    dc = 0
    tw = 0
    for _, r in df_f.iterrows():
        sc = int(r['serverCharacter'])
        cc_c = int(r['clientCharacter'])
        scards = [int(c) for c in str(r['serverCards']).split('|') if c.isdigit()]
        ccards = [int(c) for c in str(r['clientCards']).split('|') if c.isdigit()]
        sw = r['winner'] == -1
        for cid, cards, won in [(sc, scards, sw), (cc_c, ccards, not sw)]:
            if cid != char_id:
                continue
            dc += 1
            if won:
                tw += 1
            for c in set(cards):
                cc[c] += 1
                if won:
                    cw[c] += 1
    count = cc.get(card_id, 0)
    freq = 100 * count / max(1, dc)
    wr_with = 100 * cw.get(card_id, 0) / max(1, count) if count else 0
    decks_without = dc - count
    wins_without = tw - cw.get(card_id, 0)
    wr_without = 100 * wins_without / max(1, decks_without) if decks_without > 0 else 0
    wr_diff = wr_with - wr_without
    elo_with_list = []
    elo_without_list = []
    for _, r in df_f.iterrows():
        sc = int(r['serverCharacter'])
        cc_c = int(r['clientCharacter'])
        scards = [int(c) for c in str(r['serverCards']).split('|') if c.isdigit()]
        ccards = [int(c) for c in str(r['clientCards']).split('|') if c.isdigit()]
        for cid, cards, elo in [(sc, scards, r['serverUserElo']),
                                  (cc_c, ccards, r['clientUserElo'])]:
            if cid != char_id:
                continue
            if card_id in cards:
                elo_with_list.append(elo)
            else:
                elo_without_list.append(elo)
    elo_with = sum(elo_with_list) / max(1, len(elo_with_list))
    elo_without = sum(elo_without_list) / max(1, len(elo_without_list))
    copy_dist = Counter()
    for _, r in df_f.iterrows():
        sc = int(r['serverCharacter'])
        cc_c = int(r['clientCharacter'])
        scards = [int(c) for c in str(r['serverCards']).split('|') if c.isdigit()]
        ccards = [int(c) for c in str(r['clientCards']).split('|') if c.isdigit()]
        for cid, cards in [(sc, scards), (cc_c, ccards)]:
            if cid != char_id:
                continue
            c = cards.count(card_id)
            if c > 0:
                copy_dist[c] += 1
    return {
        'n_decks': dc, 'count': count, 'freq': freq,
        'wr_with': wr_with, 'wr_without': wr_without, 'wr_diff': wr_diff,
        'elo_with': elo_with, 'elo_without': elo_without,
        'copy_dist': copy_dist,
    }


def _load_char_strength():
    from pathlib import Path
    import joblib
    p = Path(__file__).parent / "models" / "character_strength.pkl"
    if p.exists():
        return joblib.load(p)
    import inicio
    return inicio.character_strength


def _get_model_and_predict():
    from Baralho_Optimal import predict_match as pm, model, cards_to_vector
    return pm, model


# ------ Comandos ------

def cmd_char(char_id):
    cs = _load_char_strength()
    name = CHARACTERS.get(char_id, f"Char {char_id}")
    strength = cs.get(char_id, 1.0)
    avg = sum(cs.values()) / len(cs)

    import pandas as pd
    from collections import defaultdict
    df = pd.read_csv('data.csv')
    mask = (df['serverUserElo'] >= 1500) & (df['clientUserElo'] >= 1500)
    df_f = df[mask]
    players = set()
    games = 0
    for _, r in df_f.iterrows():
        if int(r['serverCharacter']) == char_id:
            players.add(r['serverUserId'])
            games += 1
        if int(r['clientCharacter']) == char_id:
            players.add(r['clientUserId'])
            games += 1

    print("=" * 60)
    print(f"FORCA DA PERSONAGEM: {name}")
    print("=" * 60)
    print(f"Jogadores unicos: {len(players)}")
    print(f"Partidas analisadas: ~{games} (ELO >= 1500)")
    print(f"Forca (player-adjusted): {strength:.4f}")
    delta = (strength - 1.0) * 100
    if delta > 0:
        print(f"  -> Jogadores ganham +{delta:.1f}% MAIS com {name} do que sua media")
    else:
        print(f"  -> Jogadores ganham {delta:.1f}% MENOS com {name} do que sua media")
    print(f"Forca media das 20: {avg:.4f}")
    if strength > avg:
        print(f"  -> {name} esta ACIMA da media em {((strength/avg)-1)*100:.1f}%")
    else:
        print(f"  -> {name} esta ABAIXO da media em {((avg/strength)-1)*100:.1f}%")


def cmd_card(char_id, card_id):
    data = _load_card_data(char_id, card_id)
    card_name = get_card_name(char_id, card_id)
    char_name = CHARACTERS.get(char_id, f"Char {char_id}")

    sys_th = 1.0
    sk_th = 2.0
    threshold = sys_th if card_id <= 20 else sk_th

    freq = data['freq']
    wr_diff = data['wr_diff']
    base = freq if freq >= threshold else max(0.01, freq / 20)
    if wr_diff < -5:
        penalty = 0.3
    elif wr_diff < -2:
        penalty = 0.6
    else:
        penalty = 1.0
    weight = base * penalty

    print("=" * 60)
    print(f"CARTA: {card_name} ({char_name})")
    print("=" * 60)
    tipo = "System" if card_id <= 20 else ("Skill" if card_id <= 114 else "Spell")
    print(f"Tipo: {tipo} (ID {card_id})")
    print(f"Frequencia: {freq:.1f}% dos decks ({data['count']}/{data['n_decks']})")
    if data['copy_dist']:
        total = sum(data['copy_dist'].values())
        parts = [f"{c}x: {data['copy_dist'][c]/total*100:.1f}%" for c in sorted(data['copy_dist'])]
        print(f"  Distribuicao de copias: {', '.join(parts)}")
    print()
    print(f"Win rate COM a carta:  {data['wr_with']:.1f}%")
    print(f"Win rate SEM a carta:  {data['wr_without']:.1f}%")
    print(f"Diferenca:             {wr_diff:+.1f}%")
    if wr_diff < 0:
        print(f"  -> A carta CORRELACIONA com derrota")
    else:
        print(f"  -> A carta CORRELACIONA com vitoria")
    print()
    print(f"ELO medio COM:  {data['elo_with']:.0f}")
    print(f"ELO medio SEM:  {data['elo_without']:.0f}")
    print(f"Diferenca:       {data['elo_with'] - data['elo_without']:+.0f}")
    print()
    print(f"Peso final: {weight:.2f}")
    print(f"  freq={freq:.1f} {'>=' if freq >= threshold else '<'} threshold {threshold:.1f} "
          f"-> base = {base:.2f}")
    if penalty < 1.0:
        if wr_diff < -5:
            nivel = "< -5% → multiplicador 0.3"
        else:
            nivel = "entre -2% e -5% → multiplicador 0.6"
        print(f"  WR diff = {wr_diff:.1f}% ({nivel})")
        print(f"  {base:.2f} x {penalty} = {weight:.2f}")
    else:
        print(f"  WR diff = {wr_diff:.1f}% >= -2% → sem penalidade")


def cmd_match(s_char, c_char, s_elo, c_elo):
    pm, model = _get_model_and_predict()
    from Baralho_Optimal import cards_to_vector, random_deck
    import random as rnd

    s_name = CHARACTERS.get(s_char, f"Char {s_char}")
    c_name = CHARACTERS.get(c_char, f"Char {c_char}")

    s_opt = list(random_deck(s_char))
    c_opt = list(random_deck(c_char))

    result = pm(s_elo, c_elo, s_char, c_char, s_opt, c_opt)

    print("=" * 70)
    print(f"PREDICAO: {s_name} (ELO {s_elo}) vs {c_name} (ELO {c_elo})")
    print("=" * 70)
    print(f"Resultado: {'SERVER' if result['winner'] == 'server' else 'CLIENTE'} vence")
    print(f"  Servidor: {result['server_win_probability']:.1%}")
    print(f"  Cliente:  {result['client_win_probability']:.1%}")

    print(f"\nImpacto de cada carta no baralho do SERVIDOR ({s_name}):")
    _explain_deck(s_char, s_elo, c_char, c_elo, s_opt, c_opt, is_server=True)

    print(f"\nImpacto de cada carta no baralho do CLIENTE ({c_name}):")
    _explain_deck(c_char, c_elo, s_char, s_elo, c_opt, s_opt, is_server=False)

    print("\nFatores globais mais importantes:")
    importances = model.feature_importances_
    idxs = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:10]
    for idx in idxs:
        imp = importances[idx]
        if imp < 0.01:
            break
        if idx == 0:
            name = "char_id"
        elif 1 <= idx <= 482:
            name = "card_" + str(idx)
        elif idx == 483:
            name = "elo_diff"
        elif idx == 484:
            name = "n_unique_cards"
        elif idx == 485:
            name = "total_system"
        elif idx == 486:
            name = "total_skill"
        elif idx == 487:
            name = "total_spell"
        else:
            name = f"feat_{idx}"
        print(f"  {name:<35} {imp:.1%}")


def _explain_deck(char_id, char_elo, opp_char, opp_elo, deck, opp_deck, is_server):
    pm, model = _get_model_and_predict()
    from Baralho_Optimal import cards_to_vector, random_deck
    import random as rnd

    if is_server:
        base = pm(char_elo, opp_elo, char_id, opp_char, deck, opp_deck)
        base_prob = base['server_win_probability']
    else:
        base = pm(opp_elo, char_elo, opp_char, char_id, opp_deck, deck)
        base_prob = 1 - base['server_win_probability']

    unique = list(set(deck))
    impacts = []
    for card_id in unique:
        count = deck.count(card_id)
        without = [c for c in deck if c != card_id]
        filler = []
        available = sorted(CHAR_CARDS[char_id])
        while len(without) + len(filler) < 20:
            c = rnd.choice(available)
            if filler.count(c) < 4:
                filler.append(c)
        test_deck = without + filler

        if is_server:
            r = pm(char_elo, opp_elo, char_id, opp_char, test_deck, opp_deck)
            new_prob = r['server_win_probability']
        else:
            r = pm(opp_elo, char_elo, opp_char, char_id, opp_deck, test_deck)
            new_prob = 1 - r['server_win_probability']

        delta = new_prob - base_prob
        impacts.append((card_id, count, delta))

    impacts.sort(key=lambda x: abs(x[2]), reverse=True)
    for card_id, count, delta in impacts:
        name = get_card_name(char_id, card_id)
        direction = "+" if delta > 0 else ""
        label = "piora" if delta < 0 else "melhora"
        print(f"  {name:<45} {count}x  {direction}{delta:.1%}  ({label})")


# ------ CLI ------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python explicar.py --char <id>")
        print("  python explicar.py --card <char_id> <card_id>")
        print("  python explicar.py --match <s_char> <c_char> <s_elo> <c_elo>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "--char":
        cmd_char(int(sys.argv[2]))
    elif cmd == "--card":
        cmd_card(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "--match":
        cmd_match(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print(f"Comando desconhecido: {cmd}")
        sys.exit(1)
