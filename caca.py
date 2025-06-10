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
        
        # Grid states: 0=unknown, 1=miss, 2=hit, 3=sunk
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Cache for probability calculations
        self._prob_cache = None
        self._cache_grid_state = None

    def is_in_bounds(self, i, j):
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size

    def get_ship_cells(self, i, j, ship_len, vertical):
        """Get all cells occupied by a ship placement."""
        if vertical:
            return [(i + k, j) for k in range(ship_len)]
        else:
            return [(i, j + k) for k in range(ship_len)]

    def can_place_ship_fast(self, i, j, ship_len, vertical):
        """Fast ship placement check without complex constraint solving."""
        ship_cells = self.get_ship_cells(i, j, ship_len, vertical)
        
        # Quick bounds check
        if vertical and i + ship_len > self.grid_size:
            return False
        if not vertical and j + ship_len > self.grid_size:
            return False
            
        # Check ship cells
        for ci, cj in ship_cells:
            if self.grid[ci, cj] == 1:  # Can't place on miss
                return False
        
        # Quick adjacency check - no ships can touch diagonally
        for ci, cj in ship_cells:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = ci + di, cj + dj
                    if (self.is_in_bounds(ni, nj) and 
                        (ni, nj) not in ship_cells and 
                        self.grid[ni, nj] in [2, 3]):  # Hit or sunk
                        return False
        
        return True

    def find_hit_segments(self):
        """Find all contiguous segments of hits that need to be completed."""
        segments = []
        visited = set()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 2 and (i, j) not in visited:  # Unsunk hit
                    # Find connected hits in straight lines only
                    segment = [(i, j)]
                    visited.add((i, j))
                    
                    # Check horizontal extension
                    for direction in [1, -1]:
                        k = 1
                        while True:
                            ni, nj = i, j + direction * k
                            if (self.is_in_bounds(ni, nj) and 
                                self.grid[ni, nj] == 2 and 
                                (ni, nj) not in visited):
                                segment.append((ni, nj))
                                visited.add((ni, nj))
                                k += 1
                            else:
                                break
                    
                    # Check vertical extension
                    for direction in [1, -1]:
                        k = 1
                        while True:
                            ni, nj = i + direction * k, j
                            if (self.is_in_bounds(ni, nj) and 
                                self.grid[ni, nj] == 2 and 
                                (ni, nj) not in visited):
                                segment.append((ni, nj))
                                visited.add((ni, nj))
                                k += 1
                            else:
                                break
                    
                    segments.append(sorted(segment))
        
        return segments

    def get_valid_placements_fast(self):
        """Fast enumeration of valid ship placements."""
        placements = []
        hit_segments = self.find_hit_segments()
        
        for ship_len, count in self.ships.items():
            if count <= 0:
                continue
                
            for _ in range(count):  # For each ship of this length
                ship_placements = []
                
                # If we have hit segments, prioritize placements that cover them
                if hit_segments:
                    for segment in hit_segments:
                        segment_len = len(segment)
                        if segment_len > ship_len:
                            continue  # Ship too small for this segment
                        
                        # Try to extend segment in both directions
                        if len(segment) >= 2:
                            # Determine orientation from segment
                            is_horizontal = segment[0][0] == segment[1][0]
                        else:
                            # Single hit - try both orientations
                            is_horizontal = None
                        
                        orientations = [True, False] if is_horizontal is None else [not is_horizontal]
                        
                        for vertical in orientations:
                            # Try different positions that cover this segment
                            if vertical:
                                min_row = min(cell[0] for cell in segment)
                                max_row = max(cell[0] for cell in segment)
                                col = segment[0][1]
                                
                                # Try starting positions
                                for start_row in range(max(0, max_row - ship_len + 1), 
                                                     min(self.grid_size - ship_len + 1, min_row + 1)):
                                    if self.can_place_ship_fast(start_row, col, ship_len, True):
                                        ship_cells = self.get_ship_cells(start_row, col, ship_len, True)
                                        if all(cell in ship_cells for cell in segment):
                                            ship_placements.append((ship_len, start_row, col, True))
                            else:
                                # Horizontal
                                row = segment[0][0]
                                min_col = min(cell[1] for cell in segment)
                                max_col = max(cell[1] for cell in segment)
                                
                                for start_col in range(max(0, max_col - ship_len + 1), 
                                                     min(self.grid_size - ship_len + 1, min_col + 1)):
                                    if self.can_place_ship_fast(row, start_col, ship_len, False):
                                        ship_cells = self.get_ship_cells(row, start_col, ship_len, False)
                                        if all(cell in ship_cells for cell in segment):
                                            ship_placements.append((ship_len, row, start_col, False))
                
                # Also add placements that don't cover hit segments (for remaining ships)
                if len(ship_placements) < count:
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            for vertical in [True, False]:
                                if self.can_place_ship_fast(i, j, ship_len, vertical):
                                    ship_cells = self.get_ship_cells(i, j, ship_len, vertical)
                                    # Make sure it doesn't conflict with hit segments
                                    conflicts = False
                                    for segment in hit_segments:
                                        segment_cells = set(segment)
                                        ship_cells_set = set(ship_cells)
                                        intersection = segment_cells.intersection(ship_cells_set)
                                        if intersection and not segment_cells.issubset(ship_cells_set):
                                            conflicts = True
                                            break
                                    
                                    if not conflicts:
                                        placement = (ship_len, i, j, vertical)
                                        if placement not in ship_placements:
                                            ship_placements.append(placement)
                
                placements.extend(ship_placements[:count])  # Limit to available ships
        
        return placements

    def compute_probabilities(self):
        """Compute probability grid efficiently."""
        # Check cache
        current_state = self.grid.copy()
        if (self._prob_cache is not None and 
            self._cache_grid_state is not None and 
            np.array_equal(current_state, self._cache_grid_state)):
            return self._prob_cache.copy()
        
        prob_grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
        placements = self.get_valid_placements_fast()
        
        if not placements:
            self._prob_cache = prob_grid.copy()
            self._cache_grid_state = current_state.copy()
            return prob_grid
        
        # Count coverage for each cell
        for ship_len, i, j, vertical in placements:
            ship_cells = self.get_ship_cells(i, j, ship_len, vertical)
            for ci, cj in ship_cells:
                prob_grid[ci, cj] += 1
        
        # Normalize
        total_placements = len(placements)
        if total_placements > 0:
            prob_grid /= total_placements
        
        # Cache result
        self._prob_cache = prob_grid.copy()
        self._cache_grid_state = current_state.copy()
        
        return prob_grid

    def compute_information_gain_fast(self):
        """Fast approximation of information gain."""
        info_gain_grid = np.zeros((self.grid_size, self.grid_size))
        prob_grid = self.compute_probabilities()
        
        # Simple heuristic: information gain ≈ p * (1-p) * 4
        # This approximates the entropy reduction without expensive simulation
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] != 0:
                    continue
                    
                p = prob_grid[i, j]
                if 0 < p < 1:
                    # Heuristic: cells with p≈0.5 have highest info gain
                    # Weight by neighborhood density for better targeting
                    neighbor_prob = 0
                    neighbor_count = 0
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if self.is_in_bounds(ni, nj):
                                neighbor_prob += prob_grid[ni, nj]
                                neighbor_count += 1
                    
                    avg_neighbor_prob = neighbor_prob / neighbor_count if neighbor_count > 0 else 0
                    
                    # Combine local probability with neighborhood context
                    base_gain = 4 * p * (1 - p)  # Approximates entropy
                    neighborhood_bonus = avg_neighbor_prob * 0.5  # Bonus for high-probability neighborhoods
                    
                    info_gain_grid[i, j] = base_gain + neighborhood_bonus
        
        return info_gain_grid

    def get_next_best_move(self):
        """Get optimal next move quickly."""
        # First priority: complete existing hit segments
        hit_segments = self.find_hit_segments()
        
        if hit_segments:
            # Find cells adjacent to hit segments
            candidates = []
            for segment in hit_segments:
                if len(segment) == 1:
                    # Single hit - try all 4 directions
                    i, j = segment[0]
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (self.is_in_bounds(ni, nj) and 
                            self.grid[ni, nj] == 0):
                            candidates.append((ni, nj))
                else:
                    # Multi-hit segment - extend in line
                    if len(segment) >= 2:
                        # Determine orientation
                        is_horizontal = segment[0][0] == segment[1][0]
                        
                        if is_horizontal:
                            row = segment[0][0]
                            min_col = min(cell[1] for cell in segment)
                            max_col = max(cell[1] for cell in segment)
                            
                            # Try extending left and right
                            for col in [min_col - 1, max_col + 1]:
                                if (self.is_in_bounds(row, col) and 
                                    self.grid[row, col] == 0):
                                    candidates.append((row, col))
                        else:
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
        
        # No hit segments to complete - use information gain
        info_gain = self.compute_information_gain_fast()
        max_gain = np.max(info_gain)
        
        if max_gain > 0:
            # Find all cells with maximum gain
            max_positions = np.where(info_gain == max_gain)
            if len(max_positions[0]) > 0:
                return (max_positions[0][0], max_positions[1][0])
        
        # Fallback to highest probability
        prob_grid = self.compute_probabilities()
        max_prob = np.max(prob_grid)
        
        if max_prob > 0:
            max_positions = np.where(prob_grid == max_prob)
            if len(max_positions[0]) > 0:
                return (max_positions[0][0], max_positions[1][0])
        
        return None

    def update_cell(self, row, col, result, sunk_ship_length=None):
        """Update grid with shot result."""
        if result == 'hit':
            self.grid[row, col] = 2
        elif result == 'miss':
            self.grid[row, col] = 1
        elif result == 'sunk':
            self.grid[row, col] = 3
            
        # Invalidate cache
        self._prob_cache = None
        self._cache_grid_state = None
        
        # Handle sunk ship
        if sunk_ship_length and sunk_ship_length in self.ships:
            if self.ships[sunk_ship_length] > 0:
                self.ships[sunk_ship_length] -= 1
                self._mark_sunk_ship_area()

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