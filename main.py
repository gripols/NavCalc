#!/usr/bin/env python3
import copy
import math
import operator
from collections import namedtuple
from functools import partial, reduce, lru_cache
from itertools import chain, product
from typing import Dict, Iterator, List, Optional, Set, Tuple
from colorama import Fore, Style

import matplotlib.pyplot as plt
import numpy as np


# FIXME:
# - Никаких вложенных циклов
# - Написать документацию
# - Используйте команды--максимум 3 букв
# - Добавить Пояснения к Стратегиям


# immutable like structs
Position = namedtuple("Position", ["row", "col"])
Ship = namedtuple("Ship", ["length", "count"])
GameState = namedtuple("GameState", ["grid", "ships", "grid_size"])
ShotResult = namedtuple("ShotResult", ["position", "outcome"])

# history structs
GameAction = namedtuple("GameAction", ["position", "result", "description"])
GameHistory = namedtuple("GameHistory", ["states", "actions", "current_index"])

# cell consts
UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3

# defaults
DEFAULT_GRID_SIZE = 10  # NO TOUCH
DEFAULT_SHIPS: Dict[int, int] = {4: 1, 3: 2, 2: 3, 1: 4}


def create_empty_grid(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=int)


def is_valid_position(pos: Position, grid_size: int) -> bool:
    return 0 <= pos.row < grid_size and 0 <= pos.col < grid_size


def create_initial_state(
    grid_size: int = DEFAULT_GRID_SIZE, ships: Optional[Dict[int, int]] = None
) -> GameState:
    ships = ships or DEFAULT_SHIPS.copy()
    return GameState(
        grid=create_empty_grid(grid_size), ships=ships, grid_size=grid_size
    )


def create_initial_history(initial_state: GameState) -> GameHistory:
    return GameHistory(states=[initial_state], actions=[], current_index=0)


def add_state_to_history(
    history: GameHistory, new_state: GameState, action: GameAction
) -> GameHistory:
    states = history.states[: history.current_index + 1]
    actions = history.actions[: history.current_index + 1]  # keep them in sync
    return GameHistory(states + [new_state], actions + [action], len(states))


def can_undo(history: GameHistory) -> bool:
    return history.current_index > 0


def can_redo(history: GameHistory) -> bool:
    return history.current_index < len(history.states) - 1


def undo_state(
    history: GameHistory,
) -> Tuple[GameHistory, Optional[GameState]]:
    if not can_undo(history):
        return history, None
    new_index = history.current_index - 1
    return (
        GameHistory(history.states, history.actions, new_index),
        history.states[new_index],
    )


def redo_state(
    history: GameHistory,
) -> Tuple[GameHistory, Optional[GameState]]:
    if not can_redo(history):
        return history, None
    new_index = history.current_index + 1
    return (
        GameHistory(history.states, history.actions, new_index),
        history.states[new_index],
    )


def get_current_state(history: GameHistory) -> GameState:
    return history.states[history.current_index]


def get_last_action(history: GameHistory) -> Optional[GameAction]:
    if history.current_index == 0:
        return None
    return history.actions[history.current_index - 1]


def get_ship_positions(
    start: Position, length: int, vertical: bool
) -> List[Position]:
    return (
        [Position(start.row + i, start.col) for i in range(length)]
        if vertical
        else [Position(start.row, start.col + i) for i in range(length)]
    )


def get_adjacent_positions(
    positions: List[Position], grid_size: int
) -> Set[Position]:
    pos_set = set(positions)
    adj: Set[Position] = set()

    for p in positions:
        for dr, dc in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            q = Position(p.row + dr, p.col + dc)
            if is_valid_position(q, grid_size) and q not in pos_set:
                adj.add(q)
    return adj


def can_place_ship(
    state: GameState, start: Position, length: int, vertical: bool
) -> bool:
    ship_cells = get_ship_positions(start, length, vertical)

    # bounds + collision check
    if not all(is_valid_position(c, state.grid_size) for c in ship_cells):
        return False
    if any(state.grid[c.row, c.col] in (MISS, SUNK) for c in ship_cells):
        return False

    # no touch other ships (hits/sunk)
    for q in get_adjacent_positions(ship_cells, state.grid_size):
        if state.grid[q.row, q.col] in (HIT, SUNK):
            return False
    return True


def _expand_segment(
    state: GameState,
    frontier: List[Position],
    visited: Set[Position],
    target: Set[int],
) -> Set[Position]:
    stack: List[Position] = list(frontier)
    while stack:
        head = stack.pop()
        if head in visited:
            continue
        if state.grid[head.row, head.col] not in target:
            continue

        visited.add(head)

        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = head.row + dr, head.col + dc
            if 0 <= nr < state.grid_size and 0 <= nc < state.grid_size:
                stack.append(Position(nr, nc))

    return visited


def find_connected_positions(
    state: GameState, start: Position, target_states: Set[int]
) -> List[Position]:
    return sorted(_expand_segment(state, [start], set(), target_states))


def find_all_hit_segments(state: GameState) -> List[List[Position]]:
    segments: List[List[Position]] = []
    visited: Set[Position] = set()
    for r in range(state.grid_size):
        for c in range(state.grid_size):
            p = Position(r, c)
            if state.grid[r, c] == HIT and p not in visited:
                seg = find_connected_positions(state, p, {HIT})
                visited.update(seg)
                segments.append(seg)
    return segments


@lru_cache(maxsize=None)
def _precomputed_placements_and_adjacents(
    grid_size: int, ship_length: int
) -> List[Tuple[Tuple[Position, ...], Set[Position]]]:
    placements_with_adj = []
    for r in range(grid_size):
        for c in range(grid_size - ship_length + 1):
            placement = tuple(Position(r, c + i) for i in range(ship_length))
            adj = get_adjacent_positions(placement, grid_size)
            placements_with_adj.append((placement, adj))
    for c in range(grid_size):
        for r in range(grid_size - ship_length + 1):
            placement = tuple(Position(r + i, c) for i in range(ship_length))
            adj = get_adjacent_positions(placement, grid_size)
            placements_with_adj.append((placement, adj))
    return placements_with_adj


def generate_all_ship_placements(
    state: GameState, ship_length: int
) -> Iterator[List[Position]]:
    grid = state.grid
    size = state.grid_size

    # Compute forbidden and danger zones only once
    forbidden = {
        Position(r, c)
        for r in range(size)
        for c in range(size)
        if grid[r, c] in (MISS, SUNK)
    }
    danger = {
        Position(r, c)
        for r in range(size)
        for c in range(size)
        if grid[r, c] in (HIT, SUNK)
    }

    for placement, adjacents in _precomputed_placements_and_adjacents(
        size, ship_length
    ):
        # overlapping forbidden cells
        if any(p in forbidden for p in placement):
            continue
        # touching HIT/SUNK cells (violates non-touch rule)
        if any(a in danger for a in adjacents):
            continue
        yield list(placement)


def calculate_placement_weight(
    placement: List[Position], hit_segments: List[List[Position]]
) -> float:
    if not hit_segments:
        return 1.0

    pos_set = set(placement)
    explained_segments = 0
    partial_overlaps = 0
    total_hit_cells = sum(len(seg) for seg in hit_segments)
    explained_hit_cells = 0

    for seg in hit_segments:
        seg_set = set(seg)
        overlap = pos_set & seg_set

        if seg_set.issubset(pos_set):
            explained_segments += 1
            explained_hit_cells += len(seg)
        elif overlap:
            partial_overlaps += 1
            explained_hit_cells += len(overlap)

    if total_hit_cells == 0:
        return 1.0

    explanation_ratio = explained_hit_cells / total_hit_cells

    full_explanation_bonus = 1.0 + (explained_segments * 0.5)

    if explanation_ratio == 0.0:
        return 0.1  # Low but not zero

    return explanation_ratio * full_explanation_bonus


def compute_probability_grid_heuristic(state: GameState) -> np.ndarray:
    grid = np.zeros((state.grid_size, state.grid_size), dtype=float)
    hit_segments = find_all_hit_segments(state)

    for length, count in state.ships.items():
        if count <= 0:
            continue

        for placement in generate_all_ship_placements(state, length):
            if hit_segments and not any(
                p in placement for seg in hit_segments for p in seg
            ):
                continue

            w = calculate_placement_weight(placement, hit_segments)
            cell_weight = (w / len(placement)) * count

            rows = [p.row for p in placement]
            cols = [p.col for p in placement]
            np.add.at(grid, (rows, cols), cell_weight)

    grid[state.grid != UNKNOWN] = 0.0

    total = grid.sum()
    if total > 0:
        grid /= total

    return grid


def compute_information_gain(state: GameState) -> np.ndarray:
    base_probs = compute_probability_grid_heuristic(state)
    info_gain = np.zeros_like(base_probs)

    for r in range(state.grid_size):
        for c in range(state.grid_size):
            if state.grid[r, c] != UNKNOWN:
                continue

            pos = Position(r, c)
            p_hit = base_probs[r, c]
            p_miss = 1.0 - p_hit

            if p_hit > 0 and p_hit < 1:
                current_entropy = -(
                    p_hit * np.log2(p_hit) + p_miss * np.log2(p_miss)
                )
            else:
                current_entropy = 0.0

            info_gain[r, c] = current_entropy

    # normalize
    total = info_gain.sum()
    if total > 0:
        info_gain /= total

    return info_gain


def calculate_remaining_ship_constraints(state: GameState) -> Dict[int, float]:
    total = sum(state.ships.values())
    if total == 0:
        return {length: 0.0 for length in state.ships}
    return {length: count / total for length, count in state.ships.items()}


def apply_statistical_patterns(
    grid: np.ndarray, state: GameState
) -> np.ndarray:
    # placeholder if I want more later (fuck no)
    return grid


def get_parity_positions(grid_size: int, parity: int) -> List[Position]:
    return [
        Position(r, c)
        for r in range(grid_size)
        for c in range(grid_size)
        if (r + c) % 2 == parity
    ]


def get_optimal_parity_for_ships(ships: Dict[int, int]) -> int:
    total_ship_cells = sum(length * count for length, count in ships.items())

    single_cells = ships.get(1, 0)
    multi_cell_ships = sum(
        count for length, count in ships.items() if length > 1
    )

    if single_cells > 0 and single_cells > (multi_cell_ships * 3):
        return 1
    return 0


def build_coverage_maps(state: GameState) -> Dict[int, np.ndarray]:
    cov = {}
    for ln, cnt in state.ships.items():
        if cnt <= 0:
            continue
        m = np.zeros((state.grid_size, state.grid_size), dtype=int)
        for pl in generate_all_ship_placements(state, ln):
            for p in pl:
                m[p.row, p.col] += 1
        cov[ln] = m
    return cov


# this should fix it (I think)
def apply_parity_strategy(grid, state, boost=1.2, near_hits_boost=1.05):
    result = grid.copy()
    has_hits = np.any(state.grid == HIT)
    parity = get_optimal_parity_for_ships(state.ships)
    for r in range(state.grid_size):
        for c in range(state.grid_size):
            if state.grid[r, c] != UNKNOWN:
                continue
            if (r + c) % 2 == parity:
                result[r, c] *= boost if not has_hits else near_hits_boost
    return result


def calculate_density_multiplier(
    state: GameState,
    pos: Position,
    cov_maps: Dict[int, np.ndarray],
    window: int = 3,
) -> float:
    r0, c0 = pos
    total_weighted_density = 0.0
    total_weight = 0.0

    for ship_length, coverage_map in cov_maps.items():
        ship_count = state.ships[ship_length]
        if ship_count <= 0:
            continue

        r_start = max(0, r0 - window)
        r_end = min(state.grid_size, r0 + window + 1)
        c_start = max(0, c0 - window)
        c_end = min(state.grid_size, c0 + window + 1)

        local_region = coverage_map[r_start:r_end, c_start:c_end]

        if local_region.size == 0:
            continue

        local_density = local_region.sum() / local_region.size
        max_possible = local_region.max()

        if max_possible > 0:
            normalized_density = local_density / max_possible
        else:
            normalized_density = 0.0

        weight = ship_count * ship_length
        total_weighted_density += normalized_density * weight
        total_weight += weight

    return total_weighted_density / max(total_weight, 1.0)


def enhance_with_density(grid: np.ndarray, state: GameState) -> np.ndarray:
    cov_maps = build_coverage_maps(state)
    mult = np.zeros_like(grid)
    for r in range(state.grid_size):
        for c in range(state.grid_size):
            if state.grid[r, c] == UNKNOWN:
                mult[r, c] = calculate_density_multiplier(
                    state, Position(r, c), cov_maps
                )
    return grid * (1.0 + mult * 0.3)


def get_segment_extensions(
    segment: List[Position], state: GameState
) -> List[Position]:
    if len(segment) == 1:
        p = segment[0]
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return [
            Position(p.row + dr, p.col + dc)
            for dr, dc in dirs
            if is_valid_position(
                Position(p.row + dr, p.col + dc), state.grid_size
            )
            and state.grid[p.row + dr, p.col + dc] == UNKNOWN
        ]
    # multi cell extend linearly
    seg_sorted = sorted(segment)
    if seg_sorted[0].row == seg_sorted[1].row:  # horiz
        row = seg_sorted[0].row
        cols = sorted(p.col for p in seg_sorted)
        exts = [cols[0] - 1, cols[-1] + 1]
        return [
            Position(row, col)
            for col in exts
            if is_valid_position(Position(row, col), state.grid_size)
            and state.grid[row, col] == UNKNOWN
        ]
    else:  # vertical
        col = seg_sorted[0].col
        rows = sorted(p.row for p in seg_sorted)
        exts = [rows[0] - 1, rows[-1] + 1]
        return [
            Position(row, col)
            for row in exts
            if is_valid_position(Position(row, col), state.grid_size)
            and state.grid[row, col] == UNKNOWN
        ]


def can_continue_in_direction(
    segment: List[Position], ext: Position, target_len: int, state: GameState
) -> bool:
    seg_sorted = sorted(segment)
    if len(segment) < 2:
        return True  # single hit direction not fixed
    if seg_sorted[0].row == seg_sorted[1].row:  # horiz
        row = seg_sorted[0].row
        cols = sorted([p.col for p in seg_sorted] + [ext.col])
        needed = target_len - len(segment) - 1
        left_space = cols[0]
        right_space = state.grid_size - 1 - cols[-1]
        return left_space + right_space >= needed
    else:  # vertical
        col = seg_sorted[0].col
        rows = sorted([p.row for p in seg_sorted] + [ext.row])
        needed = target_len - len(segment) - 1
        top = rows[0]
        bottom = state.grid_size - 1 - rows[-1]
        return top + bottom >= needed


def calculate_segment_completion_probability(
    segment: List[Position], state: GameState
) -> Dict[Position, float]:
    exts = get_segment_extensions(segment, state)
    if not exts:
        return {}
    probs: Dict[Position, float] = {}
    seg_len = len(segment)
    for e in exts:
        p = 0.0
        for ship_len, ship_cnt in state.ships.items():
            if ship_cnt <= 0 or ship_len < seg_len:
                continue
            if seg_len + 1 == ship_len:
                p += 0.8  # one more hit sinks it yea
            elif seg_len + 1 < ship_len:
                if len(segment) > 1:
                    if can_continue_in_direction(segment, e, ship_len, state):
                        p += 0.4
                else:
                    p += 0.3
        probs[e] = min(p, 1.0)
    return probs


def get_prioritized_extensions(
    segment: List[Position], state: GameState
) -> List[Tuple[Position, float]]:
    return sorted(
        calculate_segment_completion_probability(segment, state).items(),
        key=lambda kv: kv[1],
        reverse=True,
    )


def compute_integrated_probability_grid(state: GameState) -> np.ndarray:
    prob_grid = compute_probability_grid_heuristic(state)

    hit_segments = find_all_hit_segments(state)

    if not hit_segments:
        prob_grid = apply_parity_strategy(
            prob_grid, state, boost=1.15, near_hits_boost=1.05
        )

        prob_grid = enhance_with_density(prob_grid, state)

    prob_grid[state.grid != UNKNOWN] = 0.0

    total = prob_grid.sum()
    if total > 0:
        prob_grid /= total

    return prob_grid


def get_move_with_uncertainty_consideration(
    state: GameState,
) -> Optional[Position]:
    hit_segments = find_all_hit_segments(state)
    for seg in hit_segments:
        extensions = get_prioritized_extensions(seg, state)
        if extensions:
            return extensions[0][0]

    prob_grid = compute_integrated_probability_grid(state)
    info_grid = compute_information_gain(state)

    combined_score = 0.7 * prob_grid + 0.3 * info_grid

    max_score = 0.0
    best_position = None

    for r in range(state.grid_size):
        for c in range(state.grid_size):
            if state.grid[r, c] == UNKNOWN and combined_score[r, c] > max_score:
                max_score = combined_score[r, c]
                best_position = Position(r, c)

    return best_position


def place_ship_on_grid(
    grid: np.ndarray, positions: List[Position]
) -> np.ndarray:
    g = grid.copy()
    for p in positions:
        g[p.row, p.col] = HIT
    return g


def mark_adjacent_as_blocked(
    grid: np.ndarray, positions: List[Position], grid_size: int
) -> np.ndarray:
    g = grid.copy()
    for q in get_adjacent_positions(positions, grid_size):
        if g[q.row, q.col] == UNKNOWN:
            g[q.row, q.col] = MISS
    return g


def update_cell(state: GameState, pos: Position, result: str) -> GameState:
    g = state.grid.copy()
    if result == "hit":
        g[pos.row, pos.col] = HIT
    elif result == "miss":
        g[pos.row, pos.col] = MISS
    elif result == "sunk":
        g[pos.row, pos.col] = SUNK
        g, ships = handle_sunk_ship(g, pos, state.ships, state.grid_size)
        return GameState(g, ships, state.grid_size)
    return GameState(g, state.ships, state.grid_size)


def handle_sunk_ship(
    grid: np.ndarray, sunk_pos: Position, ships: Dict[int, int], grid_size: int
) -> Tuple[np.ndarray, Dict[int, int]]:
    ship_cells = find_connected_positions(
        GameState(grid, ships, grid_size), sunk_pos, {HIT, SUNK}
    )
    g = grid.copy()
    for p in ship_cells:
        g[p.row, p.col] = SUNK
    g = mark_adjacent_as_blocked(g, ship_cells, grid_size)
    new_ships = ships.copy()
    ln = len(ship_cells)
    if ln in new_ships and new_ships[ln] > 0:
        new_ships[ln] -= 1
    return g, new_ships


def is_straight_line(positions: List[Position]) -> bool:
    if len(positions) <= 1:
        return True
    pos_sorted = sorted(positions)
    if all(p.row == pos_sorted[0].row for p in pos_sorted):
        cols = [p.col for p in pos_sorted]
        return all(cols[i] == cols[i - 1] + 1 for i in range(1, len(cols)))
    if all(p.col == pos_sorted[0].col for p in pos_sorted):
        rows = [p.row for p in pos_sorted]
        return all(rows[i] == rows[i - 1] + 1 for i in range(1, len(rows)))
    return False


def validate_game_state(state: GameState) -> List[str]:
    errs: List[str] = []
    all_ship_cells = [
        Position(r, c)
        for r in range(state.grid_size)
        for c in range(state.grid_size)
        if state.grid[r, c] in (HIT, SUNK)
    ]
    segments: List[List[Position]] = []
    visited: Set[Position] = set()
    for p in all_ship_cells:
        if p not in visited:
            seg = find_connected_positions(state, p, {HIT, SUNK})
            visited.update(seg)
            segments.append(seg)
    # adjacency check
    for i, s1 in enumerate(segments):
        for s2 in segments[i + 1 :]:
            if any(
                q in get_adjacent_positions(s1, state.grid_size) for q in s2
            ):
                errs.append(f"ships adjacent: {s1} vs {s2}")
    # straightness check
    for seg in segments:
        if len(seg) > 1 and not is_straight_line(seg):
            errs.append(f"ship not straight: {seg}")
    return errs


def format_positions(pos: List[Position]) -> str:
    return ", ".join(f"{chr(ord('A') + p.col)}{p.row + 1}" for p in sorted(pos))


def parse_position(s: str) -> Position:
    s = s.strip().upper()
    if len(s) < 2:
        raise ValueError("position must be at least 2 characters")
    col_char, row_str = s[0], s[1:]
    if not col_char.isalpha() or not row_str.isdigit():
        raise ValueError("invalid position format")
    col = ord(col_char) - ord("A")
    row = int(row_str) - 1
    if not (0 <= col < 10 and 0 <= row < 10):
        raise ValueError("position out of bounds")
    return Position(row, col)


def is_game_complete(state: GameState) -> bool:
    return sum(state.ships.values()) == 0


def get_remaining_ship_count(state: GameState) -> int:
    return sum(state.ships.values())


def print_grid(
    state: GameState, show_moves: bool = False, move_cnt: int = 0
) -> None:
    symbols = {
        UNKNOWN: (".", Style.RESET_ALL),
        MISS: ("M", Fore.BLUE),
        HIT: ("X", Fore.RED),
        SUNK: ("S", Fore.MAGENTA),
    }
    grid_size = state.grid_size

    header = "    " + " ".join(
        f"{chr(ord('A') + c):^3}" for c in range(grid_size)
    )
    print("\n" + header)

    separator = "   +" + "---+" * grid_size
    print(separator)

    for r in range(grid_size):
        row_label = f"{r + 1:2} |"
        row_cells = "".join(
            f" {symbols.get(state.grid[r, c], ('?', Style.RESET_ALL))[1]}{symbols.get(state.grid[r, c], ('?', Style.RESET_ALL))[0]}{Style.RESET_ALL} |"
            for c in range(grid_size)
        )
        print(row_label + row_cells)
        print(separator)

    print("\nLegend:")
    print(f"  {Style.RESET_ALL}. = Unknown")
    print(f"  {Fore.BLUE}M = Miss{Style.RESET_ALL}")
    print(f"  {Fore.RED}X = Hit{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}S = Sunk{Style.RESET_ALL}")

    remaining = [
        f"{ln}x{cnt}"
        for ln, cnt in sorted(state.ships.items(), reverse=True)
        if cnt > 0
    ]
    info = f"Remaining ships: {', '.join(remaining) if remaining else 'All ships sunk!'}"
    if show_moves:
        info += f" | Moves: {move_cnt}"
    print("\n" + info)

    if is_game_complete(state):
        print("All ships have been sunk!")


def print_history_status(history: GameHistory) -> None:
    print(
        f"history: Move {history.current_index}/{len(history.states) - 1} | "
        f"undo: {'yes' if can_undo(history) else 'no'} | "
        f"redo: {'yes' if can_redo(history) else 'no'}"
    )


def plot_heatmap(state: GameState, data: np.ndarray, title: str) -> None:
    max_val = np.max(data)
    if max_val == 0:
        print("no data to plot")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    hm = ax.imshow(
        data, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=max_val
    )
    for r in range(state.grid_size):
        for c in range(state.grid_size):
            v = data[r, c]
            if v > 0.01:
                ax.text(
                    c,
                    r,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v > max_val * 0.5 else "black",
                    fontsize=8,
                    weight="bold",
                )
    for r in range(state.grid_size):
        for c in range(state.grid_size):
            cell = state.grid[r, c]
            if cell in (MISS, HIT, SUNK):
                marker, size = {
                    MISS: ("bo", 8),
                    HIT: ("r*", 15),
                    SUNK: ("ks", 10),
                }[cell]
                ax.plot(c, r, marker, markersize=size)
    plt.colorbar(hm, label="Value")
    plt.title(f"{title} -- ships remaining: {get_remaining_ship_count(state)}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(
        range(state.grid_size),
        [chr(ord("A") + i) for i in range(state.grid_size)],
    )
    plt.yticks(
        range(state.grid_size), [str(i + 1) for i in range(state.grid_size)]
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    hist = create_initial_history(create_initial_state())
    print("BATTLESHIT")
    print("UNKNOWN: '.', MISS: 'm', HIT: 'X', SUNK: '-'\n")
    print(
        "Cmds:\n  [pos] [hit/miss/sunk] - Record shot (e.g. 'A4 hit')\n  ai - suggestion for next move\n  show - Display board\n  vld - Validate current state\n  prb - Show probability heatmap (integrated)\n  inf - Show information‑gain heatmap\n  und/red - Time‑travel moves\n  hist - List moves\n  rst - Restart game\n  man - Show help\n  quit - Exit"
    )

    while True:
        try:
            state = get_current_state(hist)
            if is_game_complete(state):
                cmd = (
                    input("all ships kaput. Type 'reset' or 'quit': ")
                    .strip()
                    .lower()
                )
                if cmd == "quit":
                    break
                if cmd == "reset":
                    hist = create_initial_history(create_initial_state())
                    continue
            inp = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nalr vro cya")
            break
        if not inp:
            continue
        parts = inp.split()
        try:
            if parts[0] == "quit":
                print("ok cya")
                break
            if parts[0] == "man":
                print(
                    "Cmds:\n  [pos] [hit/miss/sunk] - Record shot (e.g. 'A4 hit')\n  ai - suggestion for next move\n  show - Display board\n  vld - Validate current state\n  prb - Show probability heatmap (integrated)\n  inf - Show information‑gain heatmap\n  und / red - Time‑travel moves\n  hist - List moves\n  rst - Restart game\n  man - Show help\n  quit - Exit"
                )
            elif parts[0] == "und":
                if can_undo(hist):
                    hist, _ = undo_state(hist)
                    print_grid(
                        get_current_state(hist), True, hist.current_index
                    )
                    print_history_status(hist)
                else:
                    print("nothing to undo")
            elif parts[0] == "red":
                if can_redo(hist):
                    hist, _ = redo_state(hist)
                    print_grid(
                        get_current_state(hist), True, hist.current_index
                    )
                    print_history_status(hist)
                else:
                    print("nothing to redo")
            elif parts[0] == "hist":
                print("\nMove History:")
                print("  0: Initial state")
                for i, act in enumerate(hist.actions, 1):
                    marker = "*" if i == hist.current_index else " "
                    print(f"{marker} {i}: {act.description}")
            elif parts[0] == "vld":
                errs = validate_game_state(get_current_state(hist))
                print(
                    "its valid you good bro"
                    if not errs
                    else "\n".join(["Errors:"] + [f"- {e}" for e in errs])
                )
            elif parts[0] == "show":
                print_grid(get_current_state(hist), True, hist.current_index)
                print_history_status(hist)
            elif parts[0] == "rst":
                hist = create_initial_history(create_initial_state())
                print("Game reset")
            elif parts[0] == "prb":
                pg = compute_integrated_probability_grid(
                    get_current_state(hist)
                )
                plot_heatmap(
                    get_current_state(hist), pg, "Integrated Ship Probability"
                )
            elif parts[0] == "inf":
                ig = compute_information_gain(get_current_state(hist))
                plot_heatmap(
                    get_current_state(hist), ig, "Information Gain (Entropy)"
                )
            elif parts[0] == "ai":
                move = get_move_with_uncertainty_consideration(
                    get_current_state(hist)
                )
                if move is None:
                    print("no legal moves found so somethings wrong")
                    continue
                pos_str = f"{chr(ord('A') + move.col)}{move.row + 1}"
                pgrid = compute_integrated_probability_grid(
                    get_current_state(hist)
                )
                prob = pgrid[move.row, move.col]
                strategy = (
                    "following up on hits"
                    if find_all_hit_segments(get_current_state(hist))
                    else "searching"
                )
                print(
                    f"Suggested: {pos_str} | Strategy: {strategy} | Probability: {prob:.1%}"
                )
            else:
                # shot result input
                if len(parts) < 2:
                    print("usage: [pos] [hit/miss/sunk]")
                    continue
                pos_tok, res = parts[0], parts[1]
                if res not in ("hit", "miss", "sunk"):
                    print("result must be hit/miss/sunk")
                    continue
                try:
                    pos = parse_position(pos_tok)
                except ValueError as exc:
                    print(f"{exc}")
                    continue
                state = get_current_state(hist)
                if state.grid[pos.row][pos.col] != UNKNOWN:
                    print("alr shot there")
                    continue
                new_state = update_cell(state, pos, res)
                act = GameAction(
                    position=pos,
                    result=res,
                    description=f"{pos_tok.upper()} {res}",
                )
                hist = add_state_to_history(hist, new_state, act)
                print_grid(new_state, True, hist.current_index)
                print_history_status(hist)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
