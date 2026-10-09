"""Independent educational reimplementation, NOT the unpublished author's source."""
from collections import deque
import heapq
import itertools
import random
import numpy as np
from maze_helper import show_maze

_ORDER = 'NESW'
_RANDOM = False
_DELTAS = {'N':(-1,0), 'E':(0,1), 'S':(1,0), 'W':(0,-1)}

def set_order(order='NESW', random=False):
    global _ORDER, _RANDOM
    if not random and (len(order)!=4 or set(order)!=set('NESW')):
        raise ValueError('Order must be a permutation of NESW')
    _ORDER, _RANDOM = order, random
    print('Directions are checked at every step in random order.' if random else f'Directions are checked in the order {list(order)}')

def _ends(maze):
    arr = np.asarray(maze)
    starts = np.argwhere(arr == 'S'); goals = np.argwhere(arr == 'G')
    if len(starts)!=1 or len(goals)!=1: raise ValueError('Maze must have one start S and one goal G')
    return tuple(starts[0]),tuple(goals[0])

def _neighbors(maze, pos):
    directions = list(_ORDER)
    if _RANDOM: random.shuffle(directions)
    for move in directions:
        dr,dc = _DELTAS[move]
        r,c = pos[0]+dr,pos[1]+dc
        if 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1] and maze[r,c] != 'X':
            yield (r,c),move

def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
heuristic = manhattan

def _result(goal, parent, actions, reached):
    if goal is None: return dict(path=None, actions=None, reached=reached)
    path=[goal]; moves=[]
    while parent[path[-1]] is not None:
        moves.append(actions[path[-1]]);path.append(parent[path[-1]])
    path.reverse();moves.reverse()
    return dict(path=path, actions=moves, reached=reached)

def best_first_search(maze, strategy='BFS', debug=False, vis=False, W=1, animation=False):
    maze = np.asarray(maze)
    start, goal = _ends(maze)

    if strategy not in ('BFS', 'DFS', 'GBFS', 'A*'):
        raise ValueError('Unknown strategy')

    if strategy == 'DFS':
        return DFS(maze, vis=vis, animation=animation)

    parents = {start: None}
    acts = {}
    reached = {start}
    counter = itertools.count()

    if strategy == 'BFS':
        frontier = deque([start])
        pop = frontier.popleft
    else:
        frontier = []

        def score(p, g):
            return heuristic(p, goal) if strategy == 'GBFS' else g + W * heuristic(p, goal)

        heapq.heappush(frontier, (score(start, 0), -next(counter), 0, start))

    best_g = {start: 0}
    expanded = set()
    frames = []

    # Thống kê
    expanded_count = 0
    max_frontier = len(frontier)

    while frontier:
        if strategy == 'BFS':
            node = pop()
            g = best_g[node]
        else:
            _, _, g, node = heapq.heappop(frontier)

            if strategy == 'A*' and g != best_g.get(node):
                continue

        if node in expanded:
            continue

        # Kiểm tra đích trước khi mở rộng nút
        expanded.add(node)

        if animation:
            frames.append((node,))

        if node == goal:
            result = _result(node, parents, acts, expanded)

            result["expanded_count"] = expanded_count
            result["max_frontier"] = max_frontier

            if animation:
                result.update(maze=maze.copy(), frames=frames)

            if vis:
                show_path(maze, result)

            return result

        # Đếm số lần mở rộng nút
        expanded_count += 1

        for neighbor, move in _neighbors(maze, node):
            ng = g + 1

            if strategy == 'A*':
                if ng >= best_g.get(neighbor, float('inf')):
                    continue

            elif neighbor in reached:
                continue

            parents[neighbor] = node
            acts[neighbor] = move

            best_g[neighbor] = ng
            reached.add(neighbor)

            if strategy == 'BFS':
                frontier.append(neighbor)
            else:
                heapq.heappush(
                    frontier,
                    (score(neighbor, ng), -next(counter), ng, neighbor)
                )

            if strategy == 'A*' and neighbor in expanded:
                expanded.remove(neighbor)

        max_frontier = max(max_frontier, len(frontier))

    result = _result(None, parents, acts, expanded)

    result["expanded_count"] = expanded_count
    result["max_frontier"] = max_frontier

    if animation:
        result.update(maze=maze.copy(), frames=frames)

    return result

def DFS(maze, vis=False, max_tries=100000, debug_reached=False,
        check_cycle=True, limit=None, frontier_option=2, animation=False):

    maze = np.asarray(maze)
    start, goal = _ends(maze)

    stack = [(start, [start], [])]
    expanded = set()
    tries = 0
    frames = []

    # Thống kê
    expanded_count = 0
    max_frontier = len(stack)

    while stack and tries < max_tries:
        node, path, moves = stack.pop()

        tries += 1
        expanded.add(node)

        if animation:
            frames.append((node,))

        if node == goal:
            result = dict(
                path=path,
                actions=moves,
                reached=expanded,
                expanded_count=expanded_count,
                max_frontier=max_frontier
            )

            if animation:
                result.update(maze=maze.copy(), frames=frames)

            if vis:
                show_path(maze, result)

            return result

        # Không mở rộng nếu đạt giới hạn độ sâu
        if limit is not None and len(moves) >= limit:
            continue

        expanded_count += 1

        for nbr, action in _neighbors(maze, node):
            if check_cycle and nbr in path:
                continue

            stack.append((nbr, path + [nbr], moves + [action]))

        max_frontier = max(max_frontier, len(stack))

    result = dict(
        path=None,
        actions=None,
        reached=expanded,
        expanded_count=expanded_count,
        max_frontier=max_frontier
    )

    if animation:
        result.update(maze=maze.copy(), frames=frames)

    return result

def IDS(maze,frontier_option=2,max_tries=100000,vis=False):
    maze=np.asarray(maze)
    # A shortest simple path can never have more edges than the number of traversable cells minus one.
    limit_max=int(np.count_nonzero(maze!='X'))
    for limit in range(limit_max+1):
        print(f"IDS đang thử giới hạn độ sâu: {limit}")
        result=DFS(maze,limit=limit,max_tries=max_tries,frontier_option=frontier_option)
        if result['path'] is not None:
            if vis:show_path(maze,result)
            return result
    return dict(path=None,actions=None,reached=set())

def show_path(maze, result):
    path = result.get("path")

    print("Path length:", len(path) - 1 if path is not None else "No solution found")
    print("Reached squares:", len(result.get("reached", [])))

    # Hiển thị mê cung
    show_maze(maze)

    # In đường đi
    if path is not None:
        print("Path:", path)
        print("Actions:", result.get("actions", []))

def min_index(values):return min(range(len(values)),key=values.__getitem__)


def animate_maze(result, interval=90, max_frames=250):
    """Show an inline animation of expanded maze cells and the final path in Jupyter."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display

    if 'maze' not in result or 'frames' not in result:
        raise ValueError('Run search with animation=True before calling animate_maze.')
    maze = np.asarray(result['maze'])
    steps = [cell for frame in result['frames'] for cell in frame]
    # Downsample only the visualization, never the actual search result.
    stride = max(1, (len(steps) + max_frames - 1) // max_frames)
    indices = list(range(0, len(steps), stride))
    if steps and (not indices or indices[-1] != len(steps) - 1):
        indices.append(len(steps) - 1)
    fig, ax = plt.subplots(figsize=(max(5, maze.shape[1]/3), max(3, maze.shape[0]/3)))
    img = ax.imshow(np.ones((*maze.shape, 3)), interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_title('Search animation')
    path = result.get('path') or []

    def draw(frame_number):
        rgb = np.ones((*maze.shape, 3))
        rgb[maze == 'X'] = (0.15, 0.17, 0.2)
        upto = indices[frame_number] + 1 if frame_number < len(indices) else len(steps)
        for r, c in steps[:upto]:
            if maze[r, c] not in ('X', 'S', 'G'):
                rgb[r, c] = (0.8, 0.84, 0.88)
        if frame_number == len(indices):
            for r, c in path:
                if maze[r, c] not in ('S', 'G'):
                    rgb[r, c] = (1, 0.72, 0.32)
        rgb[maze == 'S'] = (0.2, 0.65, 0.3)
        rgb[maze == 'G'] = (0.85, 0.25, 0.25)
        img.set_data(rgb)
        ax.set_title(f'Expanded {upto}/{len(steps)}' if frame_number < len(indices) else 'Final path')
        return (img,)

    anim = FuncAnimation(fig, draw, frames=len(indices) + 1,
                         interval=interval, repeat=False, blit=False)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))
    return anim
