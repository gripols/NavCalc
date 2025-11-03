"""One big beautiful monolithic file."""

import sys
import os
import pickle

from collections import namedtuple
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Set, Tuple
from colorama import Fore, Style

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


Position = namedtuple("Position", ["row", "col"])
Ship = namedtuple("Ship", ["length", "count"])
GameState = namedtuple("GameState", ["grid", "ships", "grid_size"])
ShotResult = namedtuple("ShotResult", ["position", "outcome"])


GameAction = namedtuple("GameAction", ["position", "result", "description"])
GameHistory = namedtuple("GameHistory", ["states", "actions", "current_index"])

# cell consts
UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3

# defaults
DEFAULT_GRID_SIZE = 10  # NO TOUCH
DEFAULT_SHIPS: Dict[int, int] = {4: 1, 3: 2, 2: 3, 1: 4}


def create_empty_grid(size: int) -> np.ndarray:
    """Create an empty game grid of the specified size."""
    return np.zeros((size, size), dtype=int)


def is_valid_position(pos: Position, grid_size: int) -> bool:
    """Check if the given position is within grid bounds."""
    return 0 <= pos.row < grid_size and 0 <= pos.col < grid_size


def create_initial_state(
    grid_size: int = DEFAULT_GRID_SIZE, ships: Optional[Dict[int, int]] = None
) -> GameState:
    """Initialize a new GameState with empty grid and default ships."""
    if ships is None:
        ships = DEFAULT_SHIPS.copy()
    return GameState(
        grid=create_empty_grid(grid_size), ships=ships, grid_size=grid_size
    )


def create_initial_history(initial_state: GameState) -> GameHistory:
    """Create an initial history object with a single starting state."""
    return GameHistory(states=[initial_state], actions=[], current_index=0)


def add_state_to_history(
    history: GameHistory, new_state: GameState, action: GameAction
) -> GameHistory:
    """Append a new state and action to the game history timeline."""
    states = history.states[: history.current_index + 1]
    actions = history.actions[: history.current_index]  # keep them in sync
    return GameHistory(states + [new_state], actions + [action], len(states))


def can_undo(history: GameHistory) -> bool:
    """Checks if you can undo."""
    return history.current_index > 0


def can_redo(history: GameHistory) -> bool:
    """Checks if you can redo."""
    return history.current_index < len(history.states) - 1


def undo_state(
    history: GameHistory,
) -> Tuple[GameHistory, Optional[GameState]]:
    """Revert to the previous game state if undo is possible."""
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
    """Revert to the previous game state if undo is possible."""
    if not can_redo(history):
        return history, None
    new_index = history.current_index + 1
    return (
        GameHistory(history.states, history.actions, new_index),
        history.states[new_index],
    )


def get_current_state(history: GameHistory) -> GameState:
    """Return the currently active state from history."""
    return history.states[history.current_index]


def get_last_action(history: GameHistory) -> Optional[GameAction]:
    """Return the most recent GameAction from history, or None if at start."""
    if history.current_index == 0:
        return None
    return history.actions[history.current_index - 1]


def get_ship_positions(
    start: Position, length: int, vertical: bool
) -> List[Position]:
    """Generate a list of ship cell positions given a start, length, and orientation."""
    return (
        [Position(start.row + i, start.col) for i in range(length)]
        if vertical
        else [Position(start.row, start.col + i) for i in range(length)]
    )


def get_adjacent_positions(
    positions: List[Position], grid_size: int
) -> Set[Position]:
    """Return all grid positions adjacent to a set of positions."""
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
    """Check if a ship of given length can be placed at a position without violating rules."""
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
    """Expand a frontier of positions using DFS to find a connected cluster of target values."""
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
    """Find all positions connected to a start position with values in target_states."""
    return sorted(_expand_segment(state, [start], set(), target_states))


def find_all_hit_segments(state: GameState) -> List[List[Position]]:
    """Group contiguous HIT cells into distinct segments on the grid."""
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
    """Precompute legal ship placements and
    their adjacency masks for a given grid and ship length."""
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


PLACEMENT_CACHE_FILE = ".navcalc_cache.pkl"

def load_or_precompute_placements(
    grid_size: int, ships_config: Dict[int, int]
) -> Dict[int, List[Tuple[Tuple[Position, ...], Set[Position]]]]:
    """
    Loads precomputed ship placements from a cache file if it exists and
    matches the current config. Otherwise, calculates them and saves to cache.
    """
    current_config = (grid_size, tuple(sorted(ships_config.items())))

    if os.path.exists(PLACEMENT_CACHE_FILE):
        try:
            with open(PLACEMENT_CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
                if cache_data.get("config") == current_config:
                    return cache_data["placements"]
        except (IOError, pickle.PickleError, EOFError):
            # Cache is corrupt or unreadable, will recalculate.
            pass

    print("First-time setup: Pre-calculating ship placements...")
    all_placements = {}
    for ship_length in ships_config.keys():
        placements = _precomputed_placements_and_adjacents(
            grid_size, ship_length
        )
        all_placements[ship_length] = placements

    cache_to_save = {"config": current_config, "placements": all_placements}
    try:
        with open(PLACEMENT_CACHE_FILE, "wb") as f:
            pickle.dump(cache_to_save, f)
    except IOError as e:
        print(f"Warning: Could not write cache file: {e}")

    return all_placements


PRECOMPUTED_PLACEMENTS = load_or_precompute_placements(
    DEFAULT_GRID_SIZE, DEFAULT_SHIPS
)


def generate_all_ship_placements(
    state: GameState, ship_length: int
) -> Iterator[List[Position]]:
    """Yield all legal placements for a given ship length under current
    grid constraints."""
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

    for placement, adjacents in PRECOMPUTED_PLACEMENTS[ship_length]:

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
    """Evaluate how well a ship placement explains existing HIT segments."""
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
    """Compute a probability heatmap for where ships are most likely to be."""
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
    """Compute an entropy-based heatmap representing information
    gain of each shot."""
    base_probs = compute_probability_grid_heuristic(state)
    info_gain = np.zeros_like(base_probs)

    for r in range(state.grid_size):
        for c in range(state.grid_size):
            if state.grid[r, c] != UNKNOWN:
                continue

            p_hit = base_probs[r, c]
            p_miss = 1.0 - p_hit

            if 0 < p_hit < 1:
                current_entropy = -(
                    p_hit * np.log2(p_hit) + p_miss * np.log2(p_miss)
                )
            else:
                current_entropy = 0.0

            info_gain[r, c] = current_entropy

    total = info_gain.sum()
    if total > 0:
        info_gain /= total

    return info_gain


def calculate_remaining_ship_constraints(state: GameState) -> Dict[int, float]:
    """Return the relative count of remaining ships normalized to 1.0 total."""
    total = sum(state.ships.values())
    if total == 0:
        return {length: 0.0 for length in state.ships}
    return {length: count / total for length, count in state.ships.items()}


def get_parity_positions(grid_size: int, parity: int) -> List[Position]:
    """Get all grid positions with a specific parity (checkerboard pattern)."""
    return [
        Position(r, c)
        for r in range(grid_size)
        for c in range(grid_size)
        if (r + c) % 2 == parity
    ]


def get_optimal_parity_for_ships(ships: Dict[int, int]) -> int:
    """Determine which parity (even/odd) best matches the current fleet."""
    single_cells = ships.get(1, 0)
    multi_cell_ships = sum(
        count for length, count in ships.items() if length > 1
    )

    if single_cells > 0 and single_cells > (multi_cell_ships * 3):
        return 1
    return 0


def build_coverage_maps(state: GameState) -> Dict[int, np.ndarray]:
    """Build a coverage map for each ship length showing placement
    frequency per cell."""
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
    """Apply parity-based weighting only when appropriate."""
    # If 1x1 ships exist and no hits are found, parity makes no sense
    if state.ships.get(1, 0) > 0 and not np.any(state.grid == HIT):
        return grid  # Preserve uniformity — skip parity

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
    """Compute a boost multiplier based on local ship placement density."""

    def get_region(map_: np.ndarray) -> np.ndarray:
        """Return a square region around pos from the given coverage map."""
        r0 = max(0, pos[0] - window)
        r1 = min(state.grid_size, pos[0] + window + 1)
        c0 = max(0, pos[1] - window)
        c1 = min(state.grid_size, pos[1] + window + 1)
        return map_[r0:r1, c0:c1]

    weighted_sum = 0.0
    weight_total = 0.0

    for length, count in state.ships.items():
        if count <= 0:
            continue

        region = get_region(cov_maps[length])
        if region.size == 0:
            continue

        max_val = region.max()
        if max_val == 0:
            continue

        norm_density = region.sum() / (region.size * max_val)
        weight = count * length

        weighted_sum += norm_density * weight
        weight_total += weight

    return weighted_sum / weight_total if weight_total else 0.0


def enhance_with_density(grid: np.ndarray, state: GameState) -> np.ndarray:
    """Boost probability scores based on local ship density in a moving window."""
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
    """Get potential extension points for a segment to complete a ship (linearly)."""
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
    """Check if a segment could be extended to reach a full ship length
    in a given direction."""
    seg_sorted = sorted(segment)
    if len(segment) < 2:
        return True  # single hit direction not fixed

    if seg_sorted[0].row == seg_sorted[1].row:  # horiz
        cols = sorted([p.col for p in seg_sorted] + [ext.col])
        needed = target_len - len(segment) - 1
        left_space = cols[0]
        right_space = state.grid_size - 1 - cols[-1]
        return left_space + right_space >= needed

    _col = seg_sorted[0].col
    rows = sorted([p.row for p in seg_sorted] + [ext.row])
    needed = target_len - len(segment) - 1
    top = rows[0]
    bottom = state.grid_size - 1 - rows[-1]
    return top + bottom >= needed


def calculate_segment_completion_probability(
    segment: List[Position], state: GameState
) -> Dict[Position, float]:
    """Estimate the likelihood that extending a segment at each candidate will
    complete a ship."""
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
    """Return segment extension positions sorted by probability of completing
    a ship."""
    return sorted(
        calculate_segment_completion_probability(segment, state).items(),
        key=lambda kv: kv[1],
        reverse=True,
    )


def compute_integrated_probability_grid(state: GameState) -> np.ndarray:
    """Generate a normalized probability grid that
    incorporates heuristic, entropy, and strategy layers."""
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
    """Choose the next best move by combining probability and information gain
    heuristics."""
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
            if (
                state.grid[r, c] == UNKNOWN
                and combined_score[r, c] > max_score
            ):
                max_score = combined_score[r, c]
                best_position = Position(r, c)

    return best_position


def place_ship_on_grid(
    grid: np.ndarray, positions: List[Position]
) -> np.ndarray:
    """Mark all given positions on the grid as HIT and return the updated grid
    copy."""
    g = grid.copy()
    for p in positions:
        g[p.row, p.col] = HIT
    return g


def mark_adjacent_as_blocked(
    grid: np.ndarray, positions: List[Position], grid_size: int
) -> np.ndarray:
    """Mark all adjacent positions as MISS around a list of positions,
    returning a new grid."""
    g = grid.copy()
    for q in get_adjacent_positions(positions, grid_size):
        if g[q.row, q.col] == UNKNOWN:
            g[q.row, q.col] = MISS
    return g


def update_cell(state: GameState, pos: Position, result: str) -> GameState:
    """Update the grid state with a new result (hit/miss/sunk) and return
    updated GameState."""
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
    """Handle sunk ship logic: mark cells, block neighbors, and decrement
    ship count."""
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
    """Check if a list of positions form a valid straight line (horizontal
    or vertical)."""
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
    """Validate the current game grid for rule violations (adjacency, shape,
    direction)."""
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
    """Format a list of Position objects as human-readable strings like 'B5'."""
    return ", ".join(
        f"{chr(ord('A') + p.col)}{p.row + 1}" for p in sorted(pos)
    )


def parse_position(s: str) -> Position:
    """Convert user input like 'A5' into a Position object."""
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
    """Check whether all ships have been sunk."""
    return sum(state.ships.values()) == 0


def get_remaining_ship_count(state: GameState) -> int:
    """Count how many ships remain afloat."""
    return sum(state.ships.values())


def print_grid(
    state: GameState, show_moves: bool = False, move_cnt: int = 0
) -> None:
    """Prints the current game grid with symbols and move count."""
    print("   +" + "---+" * state.grid_size)

    for r in reversed(range(state.grid_size)):
        print(
            f"{r + 1:2} |"
            + "".join(
                f" {symbols(state.grid[r, c])} |"
                for c in range(state.grid_size)
            )
        )
        print("   +" + "---+" * state.grid_size)

    print(
        "    "
        + " ".join(f"{chr(ord('A') + c):^3}" for c in range(state.grid_size))
    )

    print("\nLegend:")
    for char, label, color in [
        (".", "Unknown", Style.RESET_ALL),
        ("M", "Miss", Fore.BLUE),
        ("X", "Hit", Fore.RED),
        ("S", "Sunk", Fore.MAGENTA),
    ]:
        print(f"  {color}{char} = {label}{Style.RESET_ALL}")

    remaining = ", ".join(
        f"{ln}x{cnt}"
        for ln, cnt in sorted(state.ships.items(), reverse=True)
        if cnt > 0
    )
    info = f"Remaining ships: {remaining or 'All ships kaput!'}"
    if show_moves:
        info += f" | Moves: {move_cnt}"
    print("\n" + info)

    if is_game_complete(state):
        print("All ships are kaput!")


def symbols(cell_value):
    """Return the display symbol and color for a cell value."""
    symbol_map = {
        UNKNOWN: (".", Style.RESET_ALL),
        MISS: ("M", Fore.BLUE),
        HIT: ("X", Fore.RED),
        SUNK: ("S", Fore.MAGENTA),
    }
    symbol, color = symbol_map.get(cell_value, ("?", Style.RESET_ALL))
    return f"{color}{symbol}{Style.RESET_ALL}"


def print_history_status(history: GameHistory) -> None:
    """Display undo/redo availability and current move index from history."""
    print(
        f"history: Move {history.current_index}/{len(history.states) - 1} | "
        f"undo: {'yes' if can_undo(history) else 'no'} | "
        f"redo: {'yes' if can_redo(history) else 'no'}"
    )


def plot_heatmap(state: GameState, data: np.ndarray, title: str) -> None:
    """Display a seaborn heatmap over the game grid."""
    max_val = np.max(data)
    if max_val == 0:
        print("no data to plot")
        return

    # hiding values < 0.01
    annot_data = data.copy()
    annot_data[annot_data < 0.01] = np.nan  # skips NaN annotations

    _fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        data,
        ax=ax,
        cmap="coolwarm",
        annot=annot_data,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Probability"},
        vmin=0,
        vmax=max_val if max_val > 0 else 1,
    )

    # Plot markers for MISS, HIT, SUNK on top
    for r in range(state.grid_size):
        for c in range(state.grid_size):
            cell = state.grid[r, c]
            if cell in (MISS, HIT, SUNK):
                # Use prettier, centered markers
                marker, size, color = {
                    MISS: ("o", 10, "blue"),
                    HIT: ("X", 15, "red"),
                    SUNK: ("s", 12, "black"),
                }[cell]
                ax.plot(
                    c + 0.5,  # Center cell marker (x-axis)
                    r + 0.5,  # Center cell marker (y-axis)
                    marker,
                    markersize=size,
                    color=color,
                    markeredgewidth=2,  # Make markers stand out
                )

    # labels and ticks
    plt.title(
        f"{title} -- ships remaining: {get_remaining_ship_count(state)}",
        fontsize=16,
    )
    plt.xlabel("Column")
    plt.ylabel("Row")

    # ticks to the center of the cells for Seaborn
    ax.set_xticks(np.arange(state.grid_size) + 0.5)
    ax.set_xticklabels([chr(ord("A") + i) for i in range(state.grid_size)])
    ax.set_yticks(np.arange(state.grid_size) + 0.5)
    ax.set_yticklabels([str(i + 1) for i in range(state.grid_size)])

    ax.invert_yaxis()  # keep the y-axis inverted (1 at top)

    plt.tight_layout()
    plt.show()


def print_welcome():
    """Print the game banner and list of available commands."""
    print("NavCalc")
    print("UNKNOWN: '.', MISS: 'm', HIT: 'X', SUNK: '-'\n")
    print(
        "Cmds:\n"
        "  [pos] [hit/miss/sunk] - Record shot (e.g. 'A4 hit')\n"
        "  ai    - Suggest next move\n"
        "  show  - Display board\n"
        "  vld   - Validate current state\n"
        "  prb   - Show probability heatmap\n"
        "  inf   - Show information‑gain heatmap\n"
        "  und   - Undo\n"
        "  red   - Redo\n"
        "  hist  - Move history\n"
        "  rst   - Restart game\n"
        "  man   - Show help\n"
        "  quit  - Exit"
    )


def handle_quit(_hist, _parts):
    """Handles quitting."""
    print("ok cya")
    sys.exit(0)


def handle_help(hist, _parts):
    """Prints `print_welcome` again."""
    print_welcome()
    return hist


def handle_undo(hist, _parts):
    """Undo last move if possible."""
    if can_undo(hist):
        hist, _ = undo_state(hist)
        print_grid(get_current_state(hist), True, hist.current_index)
        print_history_status(hist)
    else:
        print("nothing to undo")
    return hist


def handle_redo(hist, _parts):
    """Redo last move if possible."""
    if can_redo(hist):
        hist, _ = redo_state(hist)
        print_grid(get_current_state(hist), True, hist.current_index)
        print_history_status(hist)
    else:
        print("nothing to redo")
    return hist


def handle_hist(hist, _parts):
    """Display move history. Potential addition to this would
    be different markers so people can see them pretty clearly."""
    print("\nMove History:")
    print("  0: Initial state")
    for i, act in enumerate(hist.actions, 1):
        marker = "*" if i == hist.current_index else " "
        print(f"{marker} {i}: {act.description}")
    return hist


def handle_validate(hist, _parts):
    """Validates the current game state (adjacency, alignment, weird shapes,
    etc)."""
    errs = validate_game_state(get_current_state(hist))
    if errs:
        print("\n".join(["Errors:"] + [f"- {e}" for e in errs]))
    else:
        print("its valid you good bro")
    return hist


def handle_show(hist, _parts):
    """Prints the current game grid."""
    print_grid(get_current_state(hist), True, hist.current_index)
    print_history_status(hist)
    return hist


def handle_reset(_hist, _parts):
    """Handles resetting the game to its initial state."""
    print("Game reset")
    return create_initial_history(create_initial_state())


def handle_probability(hist, _parts):
    """Handles displaying the probability heatmap."""
    state = get_current_state(hist)
    pg = compute_integrated_probability_grid(state)
    plot_heatmap(state, pg, "Integrated Ship Probability")
    return hist


def handle_information(hist, _parts):
    """Handles displaying the info gain heatmap."""
    state = get_current_state(hist)
    ig = compute_information_gain(state)
    plot_heatmap(state, ig, "Information Gain (Entropy)")
    return hist


def handle_ai(hist, _parts):
    """Suggests the next best move."""
    state = get_current_state(hist)
    move = get_move_with_uncertainty_consideration(state)
    if move is None:
        print("no legal moves found so somethings wrong")
        return hist

    pos_str = f"{chr(ord('A') + move.col)}{move.row + 1}"
    prob = compute_integrated_probability_grid(state)[move.row, move.col]
    strategy = (
        "following up on hits" if find_all_hit_segments(state) else "searching"
    )
    print(
        f"Suggested: {pos_str} | Strategy: {strategy} | Probability: {prob:.1%}"
    )
    return hist


def handle_shot_command(hist, parts):
    """Handles user input for recording shots."""
    if len(parts) < 2:
        print_welcome()
        return hist
    pos_tok, res = parts[0], parts[1]
    if res not in ("hit", "miss", "sunk"):
        print("result must be hit/miss/sunk")
        return hist
    try:
        pos = parse_position(pos_tok)
    except ValueError as exc:
        print(f"{exc}")
        return hist

    state = get_current_state(hist)
    if state.grid[pos.row][pos.col] != UNKNOWN:
        print("alr shot there")
        return hist

    new_state = update_cell(state, pos, res)
    act = GameAction(
        position=pos, result=res, description=f"{pos_tok.upper()} {res}"
    )
    hist = add_state_to_history(hist, new_state, act)
    print_grid(new_state, True, hist.current_index)
    print_history_status(hist)
    return hist


COMMANDS = {
    "quit": handle_quit,
    "man": handle_help,
    "und": handle_undo,
    "red": handle_redo,
    "hist": handle_hist,
    "vld": handle_validate,
    "show": handle_show,
    "rst": handle_reset,
    "prb": handle_probability,
    "inf": handle_information,
    "ai": handle_ai,
}


def main():
    """Main loop of the CLI."""
    hist = create_initial_history(create_initial_state())
    print_welcome()

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
            if not inp:
                continue

            parts = inp.split()
            cmd = parts[0]

            if cmd in COMMANDS:
                hist = COMMANDS[cmd](hist, parts)
            else:
                hist = handle_shot_command(hist, parts)

        except (EOFError, KeyboardInterrupt):
            print("\nalr vro cya")
            break
        except (ValueError, IndexError) as exc:
            print(f"Error: {exc}")
        # except Exception as exc:
        #   print(f"Error: {exc}")


if __name__ == "__main__":
    main()
