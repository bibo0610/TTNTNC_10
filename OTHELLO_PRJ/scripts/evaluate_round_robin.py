"""Round-robin of five Othello agent families, using the existing paired evaluator."""
import argparse
import csv
import json
import random
from collections import Counter
from functools import partial
from itertools import combinations
from pathlib import Path
from time import perf_counter

from scripts.evaluate_cnn import evaluate, format_line, tally
from agents.cnn_agent import CNNAgent
from agents.heuristic_agent import heuristic_agent
from agents.minimax_agent import minimax_agent
from agents.mcts_agent import mcts_agent
from core.othello_engine import (
    random_agent, black, white, init_board, get_valid_moves,
    make_move, is_game_over, get_winner,
)
from history.game_history import GameHistory


def recorded_opening(seed, plies):
    rng = random.Random(seed)
    board, player = init_board(), black
    history = GameHistory()
    history.start(board)
    for _ in range(plies):
        if is_game_over(board):
            break
        legal = get_valid_moves(board, player)
        move = rng.choice(legal) if legal else None
        if move is not None:
            board = make_move(board, *move, player)
        history.record(player, move, board)
        player = -player
    return board, player, history


def evaluate_with_replays(agent, opponent, games, seed, opening_plies,
                          replay_dir, pair_number, name_a, name_b):
    """Same openings, RNG seeds and colour assignment as evaluate_cnn.evaluate."""
    results = {black: Counter(W=0, D=0, L=0),
               white: Counter(W=0, D=0, L=0)}
    openings = set()
    for i in range(games // 2):
        for colour, offset in ((black, 0), (white, 1)):
            board, player, history = recorded_opening(seed+i, opening_plies)
            openings.add(tuple(cell for row in board for cell in row))
            continuation_seed = seed + 2*i + 100000 + offset
            random.seed(continuation_seed)
            agents = {colour: agent, -colour: opponent}
            names = {colour: name_a, -colour: name_b}
            opening_turns = len(history.data['moves'])
            while not is_game_over(board):
                legal = get_valid_moves(board, player)
                move = None
                if legal:
                    move = agents[player](board, player)
                    if move not in legal:
                        raise ValueError(f'Illegal move {move!r} for {player}: {legal}')
                    board = make_move(board, *move, player)
                history.record(player, move, board)
                player = -player
            winner = get_winner(board)
            history.finish(winner)
            tally(results, colour, winner)
            history.data['tournament'] = {
                'pair': pair_number, 'seed': seed, 'opening_index': i+1,
                'opening_seed': seed+i, 'opening_turns': opening_turns,
                'continuation_seed': continuation_seed,
                'black_agent': names[black], 'white_agent': names[white],
            }
            path = replay_dir / f'pair_{pair_number:02d}_seed_{seed}_game_{2*i+offset+1:03d}.json'
            with path.open('x', encoding='utf-8') as f:
                json.dump(history.data, f, ensure_ascii=False, indent=2)
    return results, results[black] + results[white], len(openings)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/cnn_v4_1_weighted_final.pt')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--mcts-simulations', type=int, default=64)
    p.add_argument('--games-per-seed', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[2026, 3026])
    p.add_argument('--opening-plies', type=int, default=8)
    p.add_argument('--out', default='results/round_robin/round_robin_40.csv')
    p.add_argument('--save-replays', action='store_true', help='Save each completed game as a GUI-compatible JSON in game_logs/<output stem>/')
    a = p.parse_args()
    if a.games_per_seed < 2 or a.games_per_seed % 2 or not 0 <= a.opening_plies <= 50 or min(a.depth, a.mcts_simulations) < 1 or len(set(a.seeds)) != len(a.seeds):
        p.error('Use distinct seeds, positive budgets, even games-per-seed >= 2 and opening-plies in 0..50')
    out = Path(a.out)
    rank = out.with_name(out.stem + '_ranking.csv')
    if out.exists() or rank.exists():
        p.error('Output already exists; choose a new --out path')
    replay_dir = Path('game_logs') / out.stem
    if a.save_replays and replay_dir.exists():
        p.error(f'Replay directory already exists: {replay_dir}; choose a new --out stem')
    out.parent.mkdir(parents=True, exist_ok=True)
    agents = {'Random':random_agent, 'Heuristic':heuristic_agent,
              f'Minimax D{a.depth}':partial(minimax_agent, depth=a.depth),
              f'MCTS {a.mcts_simulations}':partial(mcts_agent, simulations=a.mcts_simulations),
              f'CNN {Path(a.checkpoint).stem}':CNNAgent(a.checkpoint)}
    if a.save_replays:
        replay_dir.mkdir(parents=True, exist_ok=False)
        print(f'Replays: {replay_dir}', flush=True)
    standings = {name: Counter(W=0,D=0,L=0) for name in agents}
    fields = ['agent_a','agent_b','seed','games','opening_plies','unique_openings','depth','mcts_simulations','checkpoint','wins_a','draws','losses_a','score_rate_a_pct','elapsed_seconds']
    print(f'{len(agents)} agents; 10 pairs; {a.games_per_seed * len(a.seeds)} games per pair; {10*a.games_per_seed*len(a.seeds)} games total',flush=True)
    with out.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); f.flush()
        for n,(x,y) in enumerate(combinations(agents,2),1):
            for seed in a.seeds:
                print(f'[{n}/10] {x} vs {y}; seed={seed}',flush=True)
                start=perf_counter()
                if a.save_replays:
                    _, c, unique = evaluate_with_replays(
                        agents[x], agents[y], a.games_per_seed, seed, a.opening_plies,
                        replay_dir, n, x, y)
                else:
                    _, c, unique=evaluate(agents[x],agents[y],a.games_per_seed,seed,a.opening_plies)
                elapsed=perf_counter()-start
                w,d,l=(c[k] for k in ('W','D','L'))
                assert w+d+l==a.games_per_seed
                standings[x].update(W=w,D=d,L=l)
                standings[y].update(W=l,D=d,L=w)
                writer.writerow(dict(agent_a=x,agent_b=y,seed=seed,games=a.games_per_seed,opening_plies=a.opening_plies,unique_openings=unique,depth=a.depth,mcts_simulations=a.mcts_simulations,checkpoint=a.checkpoint,wins_a=w,draws=d,losses_a=l,score_rate_a_pct=100*(w+0.5*d)/a.games_per_seed,elapsed_seconds=round(elapsed,6)))
                f.flush()
                print(format_line('TOTAL (agent A)',c),flush=True)
    order=sorted(standings,key=lambda name: standings[name]['W']+0.5*standings[name]['D'],reverse=True)
    with rank.open('x', newline='', encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['rank','agent','games','wins','draws','losses','score_points','win_rate_pct','score_rate_pct'])
        last_points=None; position=0
        print('\nFINAL STANDINGS')
        for i,name in enumerate(order,1):
            c=standings[name]; w,d,l=(c[k] for k in ('W','D','L')); n=w+d+l; points=w+0.5*d
            if points!=last_points: position=i
            last_points=points
            writer.writerow([position,name,n,w,d,l,points,100*w/n,100*points/n]); print(f'{position}. '+format_line(name,c))
    print(f'Saved {out} and {rank}')
    if a.save_replays:
        print(f'Saved {10*a.games_per_seed*len(a.seeds)} replay JSON files in {replay_dir}')


if __name__=='__main__':
    main()
