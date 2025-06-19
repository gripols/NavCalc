import unittest
import numpy as np
from statistics import mean
from main import (
    HIT, MISS, SUNK, UNKNOWN,
    GameState, GameHistory, GameAction, Position,
    create_initial_state, create_initial_history,
    update_cell, validate_game_state, is_game_complete,
    compute_probability_grid_heuristic, compute_information_gain,
    get_move_with_uncertainty_consideration, parse_position,
    get_current_state, add_state_to_history, get_remaining_ship_count
)

class TestBattleshipAI(unittest.TestCase):

    def setUp(self):
        self.state = create_initial_state()
        self.history = create_initial_history(self.state)

    def test_initial_state_valid(self):
        self.assertEqual(len(validate_game_state(self.state)), 0)
        self.assertFalse(is_game_complete(self.state))

    def test_parse_position_valid(self):
        pos = parse_position("B4")
        self.assertEqual(pos.row, 3)
        self.assertEqual(pos.col, 1)

    def test_parse_position_invalid(self):
        with self.assertRaises(ValueError):
            parse_position("Z100")

    def test_update_cell_hit(self):
        pos = Position(0, 0)
        new_state = update_cell(self.state, pos, "hit")
        self.assertEqual(new_state.grid[0, 0], HIT)

    def test_update_cell_miss(self):
        pos = Position(0, 1)
        new_state = update_cell(self.state, pos, "miss")
        self.assertEqual(new_state.grid[0, 1], MISS)

    def test_update_cell_sunk(self):
        pos = Position(2, 2)
        self.state.grid[pos.row, pos.col] = 1  # Simulate prior hit
        self.state.ships[1] = 1
        new_state = update_cell(self.state, pos, "sunk")
        self.assertEqual(new_state.grid[pos.row, pos.col], SUNK)

    def test_probability_grid_sums_to_1(self):
        prob_grid = compute_probability_grid_heuristic(self.state)
        self.assertAlmostEqual(prob_grid.sum(), 1.0, places=5)

    def test_information_gain_non_negative(self):
        info_gain = compute_information_gain(self.state)
        self.assertTrue(np.all(info_gain >= 0))

    def test_uncertainty_move_legality(self):
        move = get_move_with_uncertainty_consideration(self.state)
        self.assertIsInstance(move, Position)
        self.assertEqual(self.state.grid[move.row, move.col], 0)  # UNKNOWN

    def test_game_completion(self):
        state = GameState(
            grid=self.state.grid.copy(),
            ships={k: 0 for k in self.state.ships},
            grid_size=self.state.grid_size
        )
        self.assertTrue(is_game_complete(state))

    def test_add_and_get_history(self):
        pos = Position(1, 1)
        new_state = update_cell(self.state, pos, "miss")
        new_action = GameAction(position=pos, result="miss", description="B2 miss")
        hist = add_state_to_history(self.history, new_state, new_action)
        current_state = get_current_state(hist)
        self.assertEqual(current_state.grid[1, 1], MISS)

    def test_remaining_ship_count(self):
        total = get_remaining_ship_count(self.state)
        self.assertGreater(total, 0)

    def test_mean_move_count_benchmark(self):
        # Benchmark: mean move count must be under 57 moves
        max_allowed_mean = 57
        num_games = 500
        move_counts = []

        for _ in range(num_games):
            hist = create_initial_history(create_initial_state())
            moves = 0

            while not is_game_complete(get_current_state(hist)):
                move = get_move_with_uncertainty_consideration(get_current_state(hist))
                if move is None:
                    break
                state = get_current_state(hist)
                if state.grid[move.row, move.col] != UNKNOWN:
                    break  # o no
                result = np.random.choice(["hit", "miss"], p=[0.25, 0.75])
                new_state = update_cell(state, move, result)
                from main import GameAction
                action = GameAction(
                    position=move, result=result, description=f"{chr(65+move.col)}{move.row+1} {result}"
                )
                hist = add_state_to_history(hist, new_state, action)
                moves += 1

            move_counts.append(moves)

        mean_moves = mean(move_counts)
        print(f"Mean moves over {num_games} games: {mean_moves:.2f}")
        self.assertLess(mean_moves, max_allowed_mean, f"AI took too long to win (mean={mean_moves:.2f})")

if __name__ == "__main__":
    unittest.main()
