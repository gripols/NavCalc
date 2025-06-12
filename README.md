# Battleshit – A Heuristic Battleship Calculator

Battleshit is a command-line Battleship calculator built in Python based on GamePigeon iMessage rules. 
It uses probabilistic modeling and information theory to guide players toward the 
best possible moves, with an emphasis on fast play and clear diagnostics.

## Features

- Smart AI move suggestions based on ship placement probabilities, information gain (Shannon entropy), and hit-segment continuation
- Visual probability and information gain heatmaps using `matplotlib`:
- Checks for ship adjacency violations and illegal ship shapes (non-straight lines) 
- Flexible CLI commands for updating hits, misses, sunk ships
- Clear ASCII board rendering and move logging
- Modular code for simulation, extension, and testing

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies via pip:

```sh
pip install numpy matplotlib
````
## Usage

Run battleshit:

```sh
python poo.py
```

You'll enter an interactive shell with commands like:

```text
A4 hit         # record a hit at A4
B3 miss        # record a miss at B3
C5 sunk 3      # record a sunk ship of length 3
ai             # get next best move
prob           # show probability heatmap
info           # show information gain heatmap
show           # show ASCII board
valid          # validate the game state
rst            # reset the game
man            # show command help
q              # quit
```

## AI Strategy

1. **Complete Segments First**
   AI prioritizes extending existing HIT chains.

2. **Entropy Maximization**
   In absence of HITs, it calculates info gain via Shannon entropy:

   $$
   H(p) = -p \log_2 p - (1 - p) \log_2(1 - p)
   $$

3. **Raw Probability Targeting**
   Falls back to choosing cells most likely to contain a ship.

## Notes and Quirks

* All logic is written in modular, testable functions.
* Probability is calculated via weighted sampling of all valid ship placements.
* Display and plot code is cleanly separated from core logic.
* `Position` uses `(row, column)` convention, starting from 0.
* Input like `'A1'` corresponds to `row=0, column=0`.
* Duplicate `parse_position()` exists at the end of the script — clean this in future.

## TODO

* GUI mode with Tkinter or PyQt
* Save/load game state
* Ship randomizer for simulating full games
* Optimization via memoization / Numba

## Credits

I would like to thank [Patrick O'Neill](https://github.com/Carbocarde/battleship.git) and Lana for the inspiration.