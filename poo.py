#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict, deque, namedtuple
from typing import List, Tuple, Set, Dict, Optional, Callable, Iterator
from functools import reduce, partial
from itertools import chain, product
import operator

# data structs
Position   = namedtuple('Position', ['row', 'column'])
Ship       = namedtuple('Ship', ['length', 'count'])
GameState  = namedtuple('GameState', ['grid', 'ships', 'grid_size'])
ShotResult = namedtuple('ShotResult', ['position', 'outcome'])

UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3

DEFAULT_GRID_SIZE    = 10
DEFAULT_SHIPS        = {4: 1, 3: 2, 2: 3, 1: 4}
EXHAUSTIVE_THRESHOLD = 1000000

# pure funcs for basic ops
def create_empty_grid(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype = int)

def valid_position(pos: Position, grid_size: int) -> bool:
    return 0 <= pos.row < grid_size and 0 <= pos.column < grid_size

def create_initial_state(grid_size: int = DEFAULT_GRID_SIZE, ships: Dict[int, int] = None) -> GameState:
    ships = ships or DEFAULT_SHIPS.copy()
    return GameState(grid = create_empty_grid(grid_size), ships = ships, grid_size = grid_size)

# ship place utils
def get_ship_positions(start: Position, length: int, vertical: bool) -> List[Position]:
    if vertical:
        return [Position(start.row + i, start.column) for i in range(length)]
    else:
        return [Position(start.row, start.column + i) for i in range(length)]
    
def get_adjacent_positions(positions: List[Position], grid_size: int) -> Set[Position]:
    position_set = set(positions)
    adjacent = set()
    
    for pos in positions:
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            adj_pos = Position(pos.row + dr, pos.column + dc)
            if (valid_position(adj_pos, grid_size) and 
                adj_pos not in position_set):
                adjacent.add(adj_pos)

    return adjacent

def can_place_ship(state: GameState, start: Position, length: int, vertical: bool) -> bool:
    
    ship_positions = get_ship_positions(start, length, vertical)

    # quick bounds check
    if not all (valid_position(pos, state.grid_size) for pos in ship_positions):
        return False
        
    # check ship pos's are available
    for pos in ship_positions:
        if state.grid[pos.row, pos.column] in [MISS, SUNK]:
            return False
    
    # adj constraint
    adjacent_positions = get_adjacent_positions(ship_positions, state.grid_size)
    for pos in adjacent_positions:
        if state.grid[pos.row, pos.column] in [HIT, SUNK]:
            return False
        
    return True
        
def mark_adjacent_as_blocked(grid: np.ndarray, positions: List[Position], 
                           grid_size: int) -> np.ndarray:
    new_grid = grid.copy()
    adjacent = get_adjacent_positions(positions, grid_size)
    
    for pos in adjacent:
        if new_grid[pos.row, pos.column] == UNKNOWN:
            new_grid[pos.row, pos.column] = MISS
    
    return new_grid

# Functional approach to finding segments
def find_connected_positions(state: GameState, start: Position, 
                           target_states: Set[int]) -> List[Position]:
    def get_neighbors(pos: Position) -> Iterator[Position]:
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = Position(pos.row + dr, pos.column + dc)
            if valid_position(neighbor, state.grid_size):
                yield neighbor
    
    def expand_segment(visited: Set[Position], frontier: List[Position]) -> Set[Position]:
        if not frontier:
            return visited
        
        current = frontier[0]
        remaining_frontier = frontier[1:]
        
        if current in visited or state.grid[current.row, current.column] not in target_states:
            return expand_segment(visited, remaining_frontier)
        
        new_visited = visited | {current}
        new_neighbors = [n for n in get_neighbors(current) if n not in new_visited]
        
        return expand_segment(new_visited, remaining_frontier + new_neighbors)
    
    return sorted(expand_segment(set(), [start]))

def find_all_hit_segments(state: GameState) -> List[List[Position]]:
    visited = set()
    segments = []
    
    for row in range(state.grid_size):
        for column in range(state.grid_size):
            pos = Position(row, column)
            if (state.grid[row, column] == HIT and pos not in visited):
                segment = find_connected_positions(state, pos, {HIT})
                if segment:
                    segments.append(segment)
                    visited.update(segment)
    
    return segments

# Probability calculation functions
def generate_all_ship_placements(state: GameState, ship_length: int) -> Iterator[List[Position]]:
    for row in range(state.grid_size):
        for column in range(state.grid_size):
            for vertical in [False, True]:
                start = Position(row, column)
                if can_place_ship(state, start, ship_length, vertical):
                    yield get_ship_positions(start, ship_length, vertical)

def calculate_placement_weight(positions: List[Position], hit_segments: List[List[Position]]) -> float:
    position_set = set(positions)
    weight = 1.0
    
    for segment in hit_segments:
        segment_set = set(segment)
        overlap = position_set & segment_set
        
        if segment_set.issubset(position_set):
            weight *= 10.0  # complete explanation
        elif overlap:
            weight *= 2.0   # partial explanation
    
    # penalty for not explaining hits when they exist
    if hit_segments and not any(position_set & set(seg) for seg in hit_segments):
        weight *= 0.1
    
    return weight

def compute_probability_grid_heuristic(state: GameState) -> np.ndarray:
    prob_grid = np.zeros((state.grid_size, state.grid_size))
    hit_segments = find_all_hit_segments(state)
    total_weight = 0.0
    
    for ship_length, ship_count in state.ships.items():
        if ship_count <= 0:
            continue
        
        placements = list(generate_all_ship_placements(state, ship_length))
        
        for positions in placements:
            weight = calculate_placement_weight(positions, hit_segments) * ship_count
            
            for pos in positions:
                prob_grid[pos.row, pos.column] += weight
            
            total_weight += weight
    
    # Normalize and mask known cells
    if total_weight > 0:
        prob_grid /= total_weight
    
    # Zero out known cells
    mask = (state.grid != UNKNOWN)
    prob_grid[mask] = 0.0
    
    return prob_grid

def compute_information_gain(state: GameState) -> np.ndarray:
    prob_grid = compute_probability_grid_heuristic(state)
    info_grid = np.zeros_like(prob_grid)
    
    # Apply Shannon entropy formula
    def shannon_entropy(p: float) -> float:
        if 0 < p < 1:
            return -p * math.log2(p) - (1-p) * math.log2(1-p)
        return 0.0
    
    # Vectorized entropy calculation
    vectorized_entropy = np.vectorize(shannon_entropy)
    info_grid = vectorized_entropy(prob_grid)
    
    return info_grid

# Move generation functions
def get_segment_extensions(segment: List[Position], state: GameState) -> List[Position]:
    if len(segment) == 1:
        # Single hit - try all directions
        pos = segment[0]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        extensions = []
        
        for dr, dc in directions:
            ext_pos = Position(pos.row + dr, pos.column + dc)
            if (valid_position(ext_pos, state.grid_size) and 
                state.grid[ext_pos.row, ext_pos.column] == UNKNOWN):
                extensions.append(ext_pos)
        
        return extensions
    
    # Multi-hit segment - extend in line
    sorted_segment = sorted(segment)
    
    # Determine orientation
    if sorted_segment[0].row == sorted_segment[1].row:
        # Horizontal
        row = sorted_segment[0].row
        cols = [pos.column for pos in sorted_segment]
        min_col, max_col = min(cols), max(cols)
        
        extensions = []
        for column in [min_col - 1, max_col + 1]:
            ext_pos = Position(row, column)
            if (valid_position(ext_pos, state.grid_size) and 
                state.grid[ext_pos.row, ext_pos.column] == UNKNOWN):
                extensions.append(ext_pos)
        
        return extensions
    else:
        # Vertical
        column = sorted_segment[0].column
        rows = [pos.row for pos in sorted_segment]
        min_row, max_row = min(rows), max(rows)
        
        extensions = []
        for row in [min_row - 1, max_row + 1]:
            ext_pos = Position(row, column)
            if (valid_position(ext_pos, state.grid_size) and 
                state.grid[ext_pos.row, ext_pos.column] == UNKNOWN):
                extensions.append(ext_pos)
        
        return extensions

def find_best_move(state: GameState) -> Optional[Position]:
    # Priority 1: Complete hit segments
    hit_segments = find_all_hit_segments(state)
    if hit_segments:
        all_extensions = []
        for segment in hit_segments:
            extensions = get_segment_extensions(segment, state)
            all_extensions.extend(extensions)
        
        if all_extensions:
            return all_extensions[0]
    
    # Priority 2: Maximum information gain
    info_grid = compute_information_gain(state)
    max_info = np.max(info_grid)
    
    if max_info > 0:
        max_positions = np.where(info_grid == max_info)
        if len(max_positions[0]) > 0:
            return Position(max_positions[0][0], max_positions[1][0])
    
    # Priority 3: Highest probability
    prob_grid = compute_probability_grid_heuristic(state)
    max_prob = np.max(prob_grid)
    
    if max_prob > 0:
        max_positions = np.where(prob_grid == max_prob)
        if len(max_positions[0]) > 0:
            return Position(max_positions[0][0], max_positions[1][0])
    
    return None

# State update functions
def update_cell(state: GameState, pos: Position, result: str) -> GameState:
    new_grid = state.grid.copy()
    
    if result == 'hit':
        new_grid[pos.row, pos.column] = HIT
    elif result == 'miss':
        new_grid[pos.row, pos.column] = MISS
    elif result == 'sunk':
        new_grid[pos.row, pos.column] = SUNK
        new_grid, updated_ships = handle_sunk_ship(new_grid, pos, state.ships, state.grid_size)
        return GameState(new_grid, updated_ships, state.grid_size)
    
    return GameState(new_grid, state.ships, state.grid_size)

def handle_sunk_ship(grid: np.ndarray, sunk_pos: Position, ships: Dict[int, int], 
                    grid_size: int) -> Tuple[np.ndarray, Dict[int, int]]:
    # Find complete sunk ship
    ship_positions = find_connected_positions(
        GameState(grid, ships, grid_size), sunk_pos, {HIT, SUNK}
    )
    
    # Update grid
    new_grid = grid.copy()
    for pos in ship_positions:
        new_grid[pos.row, pos.column] = SUNK
    
    # Mark adjacent cells as misses
    new_grid = mark_adjacent_as_blocked(new_grid, ship_positions, grid_size)
    
    # Update ship count
    new_ships = ships.copy()
    ship_length = len(ship_positions)
    if ship_length in new_ships and new_ships[ship_length] > 0:
        new_ships[ship_length] -= 1
    
    return new_grid, new_ships

# Validation functions
def validate_game_state(state: GameState) -> List[str]:
    errors = []
    
    # Find all ship segments
    all_positions = []
    for row in range(state.grid_size):
        for column in range(state.grid_size):
            if state.grid[row, column] in [HIT, SUNK]:
                all_positions.append(Position(row, column))
    
    # Group into segments
    segments = []
    visited = set()
    
    for pos in all_positions:
        if pos not in visited:
            segment = find_connected_positions(state, pos, {HIT, SUNK})
            if segment:
                segments.append(segment)
                visited.update(segment)
    
    # Check adjacency rule
    for i, ship1 in enumerate(segments):
        for j, ship2 in enumerate(segments[i+1:], i+1):
            ship1_set = set(ship1)
            ship2_set = set(ship2)
            
            # Check if any positions are adjacent
            for pos1 in ship1:
                adjacent = get_adjacent_positions([pos1], state.grid_size)
                if adjacent & ship2_set:
                    errors.append(f"ships are adjacent: {format_positions(ship1)} and {format_positions(ship2)}")
                    break
    
    # Check ship shapes
    for segment in segments:
        if len(segment) > 1 and not is_straight_line(segment):
            errors.append(f"ship is not straight: {format_positions(segment)}")
    
    return errors

def is_straight_line(positions: List[Position]) -> bool:
    if len(positions) <= 1:
        return True
    
    sorted_pos = sorted(positions)
    
    # Check horizontal
    if all(pos.row == sorted_pos[0].row for pos in sorted_pos):
        cols = [pos.column for pos in sorted_pos]
        return all(cols[i] == cols[i-1] + 1 for i in range(1, len(cols)))
    
    # Check vertical
    if all(pos.column == sorted_pos[0].column for pos in sorted_pos):
        rows = [pos.row for pos in sorted_pos]
        return all(rows[i] == rows[i-1] + 1 for i in range(1, len(rows)))
    
    return False

# Utility functions
def format_positions(positions: List[Position]) -> str:
    return ", ".join(f"{chr(ord('A') + pos.column)}{pos.row + 1}" 
                    for pos in sorted(positions))

def parse_position(pos_str: str) -> Position:
    pos_str = pos_str.strip().upper()
    if len(pos_str) < 2:
        raise ValueError("position must be at least 2 chars")
    
    col_char = pos_str[0]
    row_str = pos_str[1:]
    
    if not col_char.isalpha() or not row_str.isdigit():
        raise ValueError("invalid position format")
    
    column = ord(col_char) - ord('A')
    row = int(row_str) - 1
    
    if not (0 <= column < 10 and 0 <= row < 10):
        raise ValueError("position out of bounds")
    
    return Position(row, column)

def is_game_complete(state: GameState) -> bool:
    return sum(state.ships.values()) == 0

def get_remaining_ship_count(state: GameState) -> int:
    return sum(state.ships.values())

# Display functions
def print_grid(state: GameState) -> None:
    symbols = {UNKNOWN: '.', MISS: 'o', HIT: 'x', SUNK: 's'}
    
    print("\n   " + "  ".join(chr(ord('A') + column) for column in range(state.grid_size)))
    print("  " + "---" * state.grid_size)
    
    for row in range(state.grid_size):
        row_str = f"{row + 1:2}|"
        for column in range(state.grid_size):
            cell = state.grid[row, column]
            symbol = symbols.get(cell, '?')
            row_str += f" {symbol} "
        print(row_str)
    
    # Print remaining ships
    remaining = [f"{length}×{count}" for length, count in sorted(state.ships.items(), reverse=True) 
                if count > 0]
    
    print(f"\nRemaining ships: {', '.join(remaining) if remaining else 'all ships sunk'}")
    
    if is_game_complete(state):
        print("game complete; all ships have been sunk AAHHAHAHAHAA")

def plot_heatmap(state: GameState, data_grid: np.ndarray, title: str) -> None:
    max_val = np.max(data_grid)
    if max_val == 0:
        print("no data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap = ax.imshow(data_grid, cmap='YlOrRd', interpolation='nearest', 
                       vmin=0, vmax=max_val)
    
    # Add value annotations
    for row in range(state.grid_size):
        for column in range(state.grid_size):
            val = data_grid[row, column]
            if val > 0.01:
                color = 'white' if val > max_val * 0.5 else 'black'
                ax.text(column, row, f"{val:.2f}", ha='center', va='center',
                       color=color, fontsize=8, weight='bold')
    
    # Overlay game state
    state_symbols = {MISS: ('bo', 8), HIT: ('r*', 15), SUNK: ('ks', 10)}
    for row in range(state.grid_size):
        for column in range(state.grid_size):
            cell = state.grid[row, column]
            if cell in state_symbols:
                marker, size = state_symbols[cell]
                ax.plot(column, row, marker, markersize=size)
    
    plt.colorbar(heatmap, label="Value")
    plt.title(f"{title} - Ships remaining: {get_remaining_ship_count(state)}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(range(state.grid_size), [chr(ord('A') + i) for i in range(state.grid_size)])
    plt.yticks(range(state.grid_size), [str(i + 1) for i in range(state.grid_size)])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def parse_position(pos_str):
    pos_str = pos_str.strip().upper()
    if len(pos_str) < 2:
        raise ValueError("Position too short")
    
    column_char = pos_str[0]
    row_str = pos_str[1:]
    
    if not column_char.isalpha():
        raise ValueError("Invalid columnumn")
    if not row_str.isdigit():
        raise ValueError("Invalid row")
        
    column = ord(column_char) - ord('A')
    row = int(row_str) - 1
    
    return row, column

def main():
    state = create_initial_state()
    
    print("battleshit (battleship calculator)")
    print("cmds:")
    print("  [pos] [hit/miss/sunk] [ship_length] - Record shot (e.g., 'A4 hit', 'b3 sunk 3')")
    print("  ai          - get suggestion for the next move")
    print("  show        - display ascii (current) board")
    print("  valid       - check game validity")
    print("  prob        - probability heatmap")
    print("  info        - information gain heatmap")
    print("  rst         - reset game")
    print("  man         - help.")
    print("  q           - exit")
    print()
    
    print_grid(state)
    
    while True:
        try:
            if is_game_complete(state):
                print("YOU WIN BRO")
                command = input("enter 'rst' to play again or 'q' to gtfo: ").strip().lower()
                if command == 'quit':
                    break
                elif command == 'reset':
                    state = create_initial_state()
                    print_grid(state)
                    continue
                
                command = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("alr cya\n")
            break
        
        if not command:
            continue
            
        parts = command.split()
        
        try:
            if parts[0] == 'q':
                print("cya vro <3")
                break
            
            elif parts[0] == 'man':
                print("battleshit (battleship calculator)")
                print("cmds:")
                print("  [pos] [hit/miss/sunk] [ship_length] - Record shot (e.g., 'A4 hit', 'b3 sunk 3')")
                print("  ai          - get suggestion for the next move")
                print("  show        - display ascii (current) board")
                print("  valid       - check game validity")
                print("  prob        - probability heatmap")
                print("  info        - information gain heatmap")
                print("  rst         - reset game")
                print("  man         - help.")
                print("  q           - exit")
                print()
                
            elif parts[0] == 'valid':
                errors = validate_game_state(state)
                if errors:
                    print("game state validation errors uh oh:")
                    for error in errors:
                        print(f"   : {error}")
                else:
                    print(" uu good bruh <3")
                
            elif parts[0] == 'show':
                solver.print_grid()
                
            elif parts[0] == 'rst':
                solver.grid.fill(0)
                solver.ships = solver.original_ships.copy()
                solver._prob_cache = None
                solver._cache_grid_state = None
                print("game reset")
                
            elif parts[0] == 'prob':
                prob_grid = compute_probability_grid_heuristic(state)
                if np.max(prob_grid) > 0:
                    plot_heatmap(state, prob_grid, "Ship Probability Heatmap")
                else:
                    print("No probability data available")
                
            elif parts[0] == 'info':
                info_grid = compute_information_gain(state)
                if np.max(info_grid) > 0:
                    plot_heatmap(state, info_grid, "Information Gain Heatmap")
                else:
                    print("No information gain data available")
                
            elif parts[0] == 'ai':
                move = find_best_move()
                if move:
                    pos_str = f"{chr(ord('A') + move.column)}{move.row + 1}"
                    prob_grid = compute_probability_grid_heuristic(state)
                    prob = prob_grid[move.row, move.column]
                    
                    hit_segments = find_all_hit_segments(state)
                    strategy = " following up on the hit" if hit_segments else " on the hunt"
                    
                    print(f"\n suggested move: {pos_str}")
                    print(f"  strat: {strategy}")
                    print(f"  probability: {prob:.1%}")
                else:
                    print("no valid moves available :( ")
                    
            else:
                # Parse shot result
                if len(parts) < 2:
                    print("no bro heres how 2 use: [position] [hit/miss/sunk] [optional ship_length]")
                    continue
                    
                pos_str, result = parts[0], parts[1]
                
                try:
                    pos = parse_position(pos_str)
                except ValueError as e:
                    print(f" no bro ts invalid '{pos_str}': {e}")
                    continue
                
                if state.grid[pos.row, pos.column] != UNKNOWN:
                    print(f" NO {pos_str.upper()} HAS ALR BEEN SHOT")
                    continue
                
                if result not in ['hit', 'miss', 'sunk']:
                    print("Result must be 'hit', 'miss', or 'sunk'")
                    continue
                
                state = update_cell(state, pos, result)
                
                feedback = {
                    'hit': f"HIT RECORDED @ {pos_str.upper()}",
                    'miss': f"MISS RECORDED @ {pos_str.upper()}",
                    'sunk': f"SHIP SUNK @ {pos_str.upper()}"
                }
                
                print(feedback[result])
                
                errors = validate_game_state(state)
                if errors:
                    print(" warning bruh: uu might be breaking some rules")
                    for error in errors[:2]:
                        print(f"  : {error}")
                        
                print_grid(state)
                                    
        except Exception as e:
            print(f"Error: {e}")
            print(" type 'man' for cmd use")

if __name__ == "__main__":
    main()