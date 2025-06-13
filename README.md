# Battleboi – An Overengineered Battleship Calculator

Battleboi is a command-line Battleship calculator built in Python based on GamePigeon iMessage rules. 
It uses probabilistic modeling and information theory to guide players toward the 
best possible moves, with an emphasis on fast play and clear diagnostics.

A more detailed explanation of how Battleboi works can be found on [Medium](https://medium.com/@gripols_/overengineering-battleship-with-bayesian-statistics-and-information-gain-6e7082b1b0b8)

## Requirements

- Python 3.8+

Install dependencies via pip:

```sh
pip install -r requirements.txt
````
## Usage

Run Battleboi:

```sh
python main.py
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

## Notes and Quirks

* All logic is written in modular, testable functions.
* Probability is calculated via weighted sampling of all valid ship placements.
* Display and plot code is cleanly separated from core logic.
* `Position` uses `(row, column)` convention, starting from 1.
* Input like `'A1'` corresponds to `row=1, column=1`.
* Duplicate `parse_position()` exists at the end of the script; clean this in future!

## TODO

* Undo option
* GUI mode with Tkinter or PyQt
* Save/load game state
* Ship randomizer for simulating full games
* Optimization via memoization / Numba

## Credits

I would like to thank [Patrick O'Neill](https://github.com/Carbocarde/battleship.git) and Lana for the inspiration.