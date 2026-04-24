# Pawn Wars

A king-and-pawn chess endgame practice tool. Play freely against a Minimax AI (Alpha-Beta Pruning, depth-adjustable), then see the Stockfish-correct solution afterward. Positions are loaded from a preprocessed Lichess puzzle database. All chess rules live in Prolog; Python handles search and serves the web interface.

---

## Requirements

- Python 3.10+
- SWI-Prolog 9.1.12 or newer (required for the Janus bridge)

---

## Setup

### 1. Install SWI-Prolog

**Windows:** Download and run the installer from [https://www.swi-prolog.org/download/stable](https://www.swi-prolog.org/download/stable). During installation, check the option to add SWI-Prolog to PATH.

**Mac:**
```bash
brew install swi-prolog
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

At startup the app runs four self-checks (Prolog bridge, puzzle database, legal move assertion, transposition cache). If any fail, the error message will tell you exactly what is wrong.

---

## Puzzle database

`puzzles.json` is included and ready to use. If you want to rebuild it from a fresh Lichess dump:

1. Download `lichess_db_puzzle.csv` from [https://database.lichess.org/#puzzles](https://database.lichess.org/#puzzles)
2. Place it in the project root
3. Run `python build_puzzle_db.py`

---

## Project structure

```
app.py              Flask server and game loop
engine.py           Minimax with Alpha-Beta Pruning
board.py            Board representation and Prolog bridge
pawn_wars.pl        Prolog knowledge base (all chess rules)
build_puzzle_db.py  Offline puzzle preprocessor
puzzles.json        Preprocessed Lichess endgame positions
templates/
  index.html        Single-page browser UI
```

---

## Known issue on Windows

If `janus_swi` fails to import, make sure the SWI-Prolog `bin` directory is on your system PATH and that you are using SWI-Prolog 9.1.12 or newer. The version that ships with some package managers is too old.
