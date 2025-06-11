import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional

class BattleshipSolver:
    def __init__(self, grid_size=10, ships=None):
        self.grid_size = grid_size
        self.ships = ships or {4: 1, 3: 2, 2: 3, 1: 4}
        self.original_ships = self.ships.copy()
        
        # Grid states: 0 = unknown, 1 = miss, 2 = hit, 3 = sunk
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        # track sunk ships to avoid double counting
        # self.sunk_ships = []
        
        # Cache for probability calculations
        self._prob_cache = None
        self._cache_grid_state = None
        # self._cache_ship_state = None

    def is_in_bounds(self, i, j):
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size

    def get_ship_cells(self, i, j, ship_len, vertical):
        # Get all cells occupied by a placed ship
        if vertical:
            return [(i + k, j) for k in range(ship_len)]
        else:
            return [(i, j + k) for k in range(ship_len)]

    def can_place_ship_fast(self, i, j, ship_len, vertical):
        """Fast ship placement check without complex constraint solving."""
        ship_cells = self.get_ship_cells(i, j, ship_len, vertical)

        # quick bounds check
        if not all (self.is_in_bounds(ci, cj) for ci, cj in ship_cells):
            return False
        
        # ship cells must be either unknown or hit (not miss or sunk)
        for ci, cj in ship_cells:
            if self.grid[ci, cj] == 1:  # Can't place on miss
                return False
            # also cant place on alr sunk cells unless ts the same ship
            if self.grid[ci, cj] == 3:
                return False
        
        # CHECK ADJACENCY no ships can touch diagonally or together <3
        # check all cells around the entire ship perimeter
        for ci, cj in ship_cells:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: # skip the ship cell itself
                        continue
                    ni, nj = ci + di, cj + dj
                    if (self.is_in_bounds(ni, nj) and 
                        (ni, nj) not in ship_cells):
                        if self.grid[ni, nj] in [2, 3]:  # Hit or sunk
                            return False
        
        return True

    def find_hit_segments(self):
        # all adjacent segments of hits that need to be completed
        segments = []
        visited = set()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 2 and (i, j) not in visited:  # Unsunk hit
                    segment = self._get_connected_hits(i, j, visited)
                    if segment:
                        segment.append(sorted(segment))
        return segments

    def _get_connected_hits(self, start_i, start_j, visited):
        segment = [(start_i, start_j)]
        visited.add((start_i, start_j))

        horizontal_segment = [(start_i, start_j)]
        for direction in [1, -1]:
            k = 1
            while True:
                ni, nj = start_i, start_j + direction * k
                if (self.is_in_bounds(ni, nj) and
                    self.grid[ni, nj] == 2 and
                    (ni, nj) not in visited):
                    horizontal_segment.append((ni, nj))
                    visited.add((ni, nj))
                    k += 1
                else:
                    break
        
        # check if part of a vertical line
        vertical_segment = [(start_i, start_j)]
        visited_vertical = {(start_i, start_j)}
        for direction in [1, -1]:
            k = 1
            while True:
                ni, nj = start_i + direction * k, start_j
                if (self.is_in_bounds(ni, nj) and 
                    self.grid[ni, nj] == 2 and 
                    (ni, nj) not in visited):
                    vertical_segment.append((ni, nj))
                    visited_vertical.add((ni, nj))
                    k += 1
                else:
                    break

        # choose longer segment as ships are straight lines
        if len(horizontal_segment) > len(vertical_segment):
            return horizontal_segment
        else:
            visited.update(visited_vertical)
            return vertical_segment

    def get_valid_placements(self):
        # get all valid ways to place remaining ships
        placements = []
        hit_segments = self.find_hit_segments()
        
        # for each ship type find valid placements
        for ship_len, count in self.ships.items():
            if count <= 0:
                continue

            ship_placements = []

            # tries all positions and orientations
            # TODO: nesting is not good
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for vertical in [True, False]:
                        if self.can_place_ship_fast(i, j, ship_len, vertical):
                            ship_cells = self.get_ship_cells(i, j, ship_len, vertical)
                            
                            # check if placement is consistent w hits
                            is_valid = True
                            covers_segment = True
                            
                        for segment in hit_segments:
                            segment_set = set(segment)
                            ship_set = set(ship_cells)
                            intersection = segment_set.intersection(ship_set)

                            if intersection:
                                # ts (this ship) overlaps w a hit
                                if not segment_set.issubset(ship_set):
                                    is_valid = False
                                    break
                                covers_segment = True

                        if is_valid:
                            ship_placements.append((ship_len, i, j, vertical))

            # add valid placements for ts
            placements.extend(ship_placements)

        return placements

    def compute_probabilities(self):
        # calculates probability that each cell contains (part of) a ship
        # Check cache
        current_state = self.grid.copy()
        if (self._prob_cache is not None and 
            self._cache_grid_state is not None and 
            self._cache_ship_state is not None and 
            current_state == (self._cache_grid_state, self._cache_ship_state)):
            return self._prob_cache.copy()
        
        prob_grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
        placements = self.get_valid_placements()
        
        if not placements:
            self._prob_cache = prob_grid.copy()
            self._cache_grid_state = current_state.copy()
            self._cache_ship_state = tuple(sorted(self.ships.items()))
            return prob_grid
        
        # Count placements cover for each cell
        for ship_len, i, j, vertical in placements:
            ship_cells = self.get_ship_cells(i, j, ship_len, vertical)
            for ci, cj in ship_cells:
                prob_grid[ci, cj] += 1
        
        # Normalize by t num of placements
        total_placements = len(placements)
        if total_placements > 0:
            prob_grid /= total_placements
        
        # Cache result
        self._prob_cache = prob_grid.copy()
        self._cache_grid_state = current_state.copy()
        self._cache_ship_state = tuple(sorted(self.ships.items()))
        
        return prob_grid

    def compute_information_gain(self):
        # approx info gain for each sell
        info_gain_grid = np.zeros((self.grid_size, self.grid_size))
        prob_grid = self.compute_probabilities()
        
        # approximates entropy reduction w/o expensive simulation
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] != 0:    # skip known cells
                    continue
                    
                p = prob_grid[i, j]
                if 0 < p < 1:
                    # info gain from ts cell
                    # entropy: -p * log(p) - (1 - p) * log(1 - p)
                    entropy = -p * math.log2(p + 1e-10) - (1 - p) * math.log2(1-p + 1e-10)

                    # weight by proximity to existing hits
                    hit_bonus = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (self.is_in_bounds(ni, nj) and
                                self.grid[ni, nj] == 2): # adj to hit
                                hit_bonus += 0.5

                    info_gain_grid[i, j] = entropy + hit_bonus
        
        return info_gain_grid

    def get_next_best_move(self):
        # get best next move
        if self.is_game_complete():
            return None # no shit

        # priority 1 complete existing hit segments
        hit_segments = self.find_hit_segments()
        
        if hit_segments:
            # find cells adj to hit segments
            candidates = []
            for segment in hit_segments:
                if len(segment) == 1:
                    # singe hit so try all 4 directions
                    i, j = segment[0]
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (self.is_in_bounds(ni, nj) and 
                            self.grid[ni, nj] == 0):
                            candidates.append((ni, nj))
                else:
                    # multi hit so extend in line
                    segment_sorted = sorted(segment)
                        
                    # determine if hori or vert
                    if segment_sorted[0][0] == segment_sorted[1][0]:
                        # hori
                        row = segment_sorted[0][0]
                        min_col = min(cell[1] for cell in segment)
                        max_col = max(cell[1] for cell in segment)
                            
                        # Try extending left and right
                        for col in [min_col - 1, max_col + 1]:
                            if (self.is_in_bounds(row, col) and 
                                self.grid[row, col] == 0):
                                candidates.append((row, col))
                        else:
                            # vert seg   
                            col = segment[0][1]
                            min_row = min(cell[0] for cell in segment)
                            max_row = max(cell[0] for cell in segment)
                            
                            # Try extending up and down
                            for row in [min_row - 1, max_row + 1]:
                                if (self.is_in_bounds(row, col) and 
                                    self.grid[row, col] == 0):
                                    candidates.append((row, col))
            
            if candidates:
                return candidates[0]  # Return first valid extension
        
        # use info gain for exploration
        info_gain = self.compute_information_gain_fast()
        max_gain = np.max(info_gain)
        
        if max_gain > 0:
            max_positions = np.where(info_gain == max_gain)
            if len(max_positions[0]) > 0:
                return (max_positions[0][0], max_positions[1][0])
        
        # priority 3 is fallback to highest probability
        prob_grid = self.compute_probabilities()
        max_prob = np.max(prob_grid)
        
        if max_prob > 0:
            max_positions = np.where(prob_grid == max_prob)
            if len(max_positions[0]) > 0:
                return (max_positions[0][0], max_positions[1][0])
        
        return None

    def update_cell(self, row, col, result):
        # update w/ shot result
        if result == 'hit':
            self.grid[row, col] = 2
        elif result == 'miss':
            self.grid[row, col] = 1
        elif result == 'sunk':
            self.grid[row, col] = 3
            
        # invalidate cache
        self._invalidate_cache()

    def _handle_sunk_ship(self, sunk_row, sunk_col):
        # handle when ship is sunk so find the complete ship and update state
        # also find all connected hits/sunk cells that form the ship
        ship_cells = self._find_complete_ship(sunk_row, sunk_col)
        
        if ship_cells:
            for i, j in ship_cells:
                self.grid[i, j] = 3

            ship_length = len(ship_cells)
            if ship_length in self.ships and self.ships[ship_length] > 0:
                self.ships[ship_length] -= 1
                print(f"sunk ship of length {ship_length}")
            
            # mark diagonal cells as miss as ships cant touch diagonally
            self._mark_diagonal_misses(ship_cells)

            # store sunk ship
            self.sunk_ships.append((ship_length, ship_cells))

            """
            STOP HERE 2025-06-11 13:17
            """
    def _mark_sunk_ship_area(self):
        """Mark cells around sunk ships as misses."""
        # Find all hit/sunk cells and mark their diagonals
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] in [2, 3]:  # Hit or sunk
                    # Mark diagonal neighbors as misses
                    for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ni, nj = i + di, j + dj
                        if (self.is_in_bounds(ni, nj) and 
                            self.grid[ni, nj] == 0):
                            self.grid[ni, nj] = 1

    def print_grid(self):
        """Print current grid state."""
        print("\n   " + "  ".join([chr(ord('A') + col) for col in range(self.grid_size)]))
        for i in range(self.grid_size):
            row_str = f"{i + 1:2} "
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    c = '.'
                elif self.grid[i, j] == 1:
                    c = 'X'  # Miss
                elif self.grid[i, j] == 2:
                    c = 'H'  # Hit
                elif self.grid[i, j] == 3:
                    c = 'S'  # Sunk
                else:
                    c = '?'
                row_str += f" {c} "
            print(row_str)
        
        # Print remaining ships
        remaining = [f"{length}×{count}" for length, count in self.ships.items() if count > 0]
        print(f"\nRemaining ships: {', '.join(remaining) if remaining else 'None'}")

    def plot_heatmap(self, data_grid, title="Heatmap"):
        """Plot a heatmap efficiently."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Only show non-zero values
        display_data = data_grid.copy()
        display_data[display_data == 0] = np.nan
        
        heatmap = ax.imshow(display_data, cmap='hot', interpolation='nearest')
        
        # Add values only for significant cells
        max_val = np.nanmax(display_data) if not np.isnan(display_data).all() else 1
        threshold = max_val * 0.1
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = data_grid[i, j]
                if val > threshold:
                    color = 'white' if val > max_val * 0.6 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                           color=color, fontsize=9, weight='bold')
        
        # Mark current grid state
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:  # Miss
                    ax.plot(j, i, 'bx', markersize=12, markeredgewidth=3)
                elif self.grid[i, j] in [2, 3]:  # Hit or sunk
                    ax.plot(j, i, 'r+', markersize=15, markeredgewidth=4)
        
        plt.colorbar(heatmap, label="Value")
        plt.title(title)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.xticks(range(self.grid_size), [chr(ord('A') + i) for i in range(self.grid_size)])
        plt.yticks(range(self.grid_size), [str(i + 1) for i in range(self.grid_size)])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

def parse_position(pos_str):
    """Parse position string like 'A4' into (row, col) indices."""
    pos_str = pos_str.strip().upper()
    if len(pos_str) < 2:
        raise ValueError("Position too short")
    
    col_char = pos_str[0]
    row_str = pos_str[1:]
    
    if not col_char.isalpha():
        raise ValueError("Invalid column")
    if not row_str.isdigit():
        raise ValueError("Invalid row")
        
    col = ord(col_char) - ord('A')
    row = int(row_str) - 1
    
    return row, col

def main():
    solver = BattleshipSolver()
    
    print("=== Fast Battleship Solver ===")
    print("Commands:")
    print("  [pos] [hit/miss/sunk] [ship_length] - Record shot (e.g., 'A4 hit', 'B5 sunk 3')")
    print("  suggest     - Get AI suggestion")
    print("  show        - Display board")
    print("  prob        - Show probability heatmap")
    print("  info        - Show information gain heatmap")
    print("  reset       - Reset game")
    print("  quit        - Exit")
    print()
    
    while True:
        try:
            command = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not command:
            continue
            
        parts = command.split()
        
        try:
            if parts[0] == 'quit':
                print("Goodbye!")
                break
                
            elif parts[0] == 'show':
                solver.print_grid()
                
            elif parts[0] == 'reset':
                solver.grid.fill(0)
                solver.ships = solver.original_ships.copy()
                solver._prob_cache = None
                solver._cache_grid_state = None
                print("Game reset!")
                
            elif parts[0] == 'prob':
                prob_grid = solver.compute_probabilities()
                solver.plot_heatmap(prob_grid, "Ship Probability Heatmap")
                
            elif parts[0] == 'info':
                info_grid = solver.compute_information_gain_fast()
                solver.plot_heatmap(info_grid, "Information Gain Heatmap")
                
            elif parts[0] == 'suggest':
                move = solver.get_next_best_move()
                if move:
                    row, col = move
                    pos_str = f"{chr(ord('A') + col)}{row + 1}"
                    prob_grid = solver.compute_probabilities()
                    prob = prob_grid[row, col] if prob_grid[row, col] > 0 else 0
                    print(f"Suggested move: {pos_str} (probability: {prob:.3f})")
                else:
                    print("No valid moves available!")
                    
            else:
                # Parse shot result
                if len(parts) < 2:
                    print("Usage: [position] [hit/miss/sunk] [optional ship_length]")
                    continue
                    
                pos_str, result = parts[0], parts[1]
                ship_length = int(parts[2]) if len(parts) > 2 else None
                
                row, col = parse_position(pos_str)
                
                if not (0 <= row < solver.grid_size and 0 <= col < solver.grid_size):
                    print("Position out of bounds!")
                    continue
                    
                if result not in ['hit', 'miss', 'sunk']:
                    print("Result must be 'hit', 'miss', or 'sunk'")
                    continue
                    
                solver.update_cell(row, col, result, ship_length)
                print(f"Recorded {result} at {pos_str.upper()}")
                
                if ship_length:
                    print(f"Ship of length {ship_length} removed from fleet")
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()