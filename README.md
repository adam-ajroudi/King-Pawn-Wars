# Pawn Wars — Python + Prolog Hybrid Web App

A playable two-player chess variant containing only Kings and Pawns.
The human plays White; the AI plays Black using **Minimax with Alpha-Beta Pruning**.
All game rules and legal-move generation live entirely in **Prolog**.
Python handles the AI search engine and web server.
The browser UI uses **chessboard.js** — pieces are draggable, legal destinations
glow green, and the board updates automatically after every AI move.

---

## Prerequisites

| Requirement | Version | Download |
|---|---|---|
| SWI-Prolog | **9.1.12 or newer** | https://www.swi-prolog.org/Download.html |
| Python | 3.9 or newer | https://www.python.org/downloads/ |

Make sure `swipl` is on your system PATH after installing SWI-Prolog.
Verify with: `swipl --version`

---

## Setup & Run

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open your browser
#    http://localhost:5000
```

No other configuration is needed.

---

## How to Play

- **Drag** any white piece onto the board.
- Legal destination squares **glow green** as soon as you pick up a piece.
- Drop the piece on a legal square to confirm your move.
- The AI (Black) responds automatically; its move animates on the board.
- Click **New Game** to reset. Use the **Search depth** selector to make the AI
  stronger (depth 4 is the default; depth 5–6 is noticeably stronger but slower).

### Win conditions (enforced by Prolog)
- A pawn promotes — white reaches row 8, or black reaches row 1.
- A side loses all its pawns.

### Starting position
```
8  . . . . k . . .    black king  e8
7  p . . p . . . p    black pawns a7, d7, h7
   …
2  P . . P . . . P    white pawns a2, d2, h2
1  . . . . K . . .    white king  e1
   a b c d e f g h
```

---

## Alpha-Beta Pruning Analysis

Click **Compare AB on vs off** in the sidebar to run the Minimax search twice
from the current position — once with Alpha-Beta Pruning and once without —
and see how many nodes each version evaluates.  
These numbers can be copied directly into the project report.

---

## File Overview

| File | Purpose |
|---|---|
| `pawn_wars.pl` | Prolog knowledge base — all game rules, move generation, win conditions |
| `board.py` | Board representation (dict), Janus bridge, FEN conversion |
| `engine.py` | Minimax + Alpha-Beta, move cache, evaluation function, node counter |
| `app.py` | Flask web server — four REST endpoints |
| `templates/index.html` | Browser UI — chessboard.js, jQuery, game controls |
| `requirements.txt` | Python package list |

---

## Architecture Notes

### Bridge
Python talks to Prolog via **Janus** (`janus_swi`), which embeds SWI-Prolog
in-process. SWI-Prolog 10.0.2 was used during development.

### Strict rendering rule
`chessboard.js` receives a **FEN string** from the Python server and renders
the board.  No chess library is used for logic, legal-move generation, or
state management.  All of that lives in `pawn_wars.pl`.

### King move limitation (documented intentional design)
King moves do **not** check whether the destination square is attacked.
This prevents a circular dependency in Prolog (move generation would call
itself to determine attacked squares).  The AI never makes a suicidal king
move in practice because Minimax sees the resulting capture at the next ply
and scores it as catastrophic.  This is *emergent self-preservation through
look-ahead*, not a hardcoded constraint, and should be described as such in
the project report.

### Move cache
Before querying Prolog for legal moves at any node in the search tree, the
engine checks a Python dictionary keyed by board state and colour.  On a
cache hit the Prolog bridge is not called.  This prevents bridge performance
collapse during deep recursion.
