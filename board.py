"""
board.py  –  Pawn Wars board representation and Prolog bridge.

The board is a plain Python dict mapping (col, row) -> (color, piece_type).
  col  : int 1–8   (a=1, b=2, …, h=8)
  row  : int 1–8
  color      : str  "white" | "black"
  piece_type : str  "pawn"  | "king"

All game logic and legal-move generation live in pawn_wars.pl.
This module is the only place that talks to Prolog via Janus.
python-chess is NEVER used here; it is used only in the notebook for
rendering.
"""

import os
import sys
import janus_swi as janus

# ── Path to the Prolog knowledge base ───────────────────────────────────────
_PL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pawn_wars.pl")

# ── Bootstrap: load KB and run connectivity test ────────────────────────────

def _load_kb() -> None:
    """Consult the Prolog knowledge base exactly once."""
    # Use forward slashes so SWI-Prolog accepts the path on Windows.
    pl_path = _PL_FILE.replace("\\", "/")
    result = janus.query_once(f"consult('{pl_path}')")
    if not result:
        sys.exit(f"[FATAL] Could not consult {_PL_FILE}. Exiting.")


def connectivity_test() -> None:
    """
    Run a trivial Prolog query to confirm the bridge is working.
    Called automatically at module import time.  Exits immediately with a
    clear error message if the bridge is broken, rather than failing silently
    later during search.
    """
    try:
        result = janus.query_once("opponent(white, X)")
        if result.get("X") != "black":
            raise RuntimeError("Unexpected result from test query.")
        print("[Prolog bridge] Connectivity OK – using Janus with SWI-Prolog 10.0.2")
    except Exception as exc:
        sys.exit(f"[FATAL] Prolog bridge connectivity test failed: {exc}")


_load_kb()
connectivity_test()


# ── Column ↔ letter helpers ──────────────────────────────────────────────────

_COL_TO_LETTER = {i: chr(ord("a") + i - 1) for i in range(1, 9)}
_LETTER_TO_COL = {v: k for k, v in _COL_TO_LETTER.items()}


def col_to_letter(col: int) -> str:
    return _COL_TO_LETTER[col]


def letter_to_col(letter: str) -> int:
    return _LETTER_TO_COL[letter.lower()]


# ── Initial board setup ──────────────────────────────────────────────────────

def initial_board() -> dict:
    """
    Return the starting board for Pawn Wars.
    Each side has three pawns and one king.

    White: King e1 (5,1)  |  Pawns a2 (1,2), d2 (4,2), h2 (8,2)
    Black: King e8 (5,8)  |  Pawns a7 (1,7), d7 (4,7), h7 (8,7)
    """
    board = {}
    # White pieces
    board[(5, 1)] = ("white", "king")
    board[(1, 2)] = ("white", "pawn")
    board[(4, 2)] = ("white", "pawn")
    board[(8, 2)] = ("white", "pawn")
    # Black pieces
    board[(5, 8)] = ("black", "king")
    board[(1, 7)] = ("black", "pawn")
    board[(4, 7)] = ("black", "pawn")
    board[(8, 7)] = ("black", "pawn")
    return board


# ── Python → Prolog term conversion ─────────────────────────────────────────

def board_to_prolog_str(board: dict) -> str:
    """
    Convert the Python board dict into a Prolog list term string suitable for
    direct embedding inside a query goal string sent through Janus.

    Example output:
        [piece(white,king,5,1),piece(white,pawn,1,2),...]
    """
    pieces = [
        f"piece({color},{ptype},{col},{row})"
        for (col, row), (color, ptype) in board.items()
    ]
    return "[" + ",".join(pieces) + "]"


# ── Legal-move query ─────────────────────────────────────────────────────────

def get_legal_moves(board: dict, color: str) -> list:
    """
    Query Prolog for all legal moves available to *color* on *board*.
    Uses legal_moves_py/3, which returns each move as a [FC, FR, TC, TR] list
    so that Janus can map it to a Python list without compound-term conversion.
    Returns a list of (from_col, from_row, to_col, to_row) tuples.
    """
    board_str = board_to_prolog_str(board)
    goal = f"legal_moves_py({board_str}, {color}, Moves)"
    result = janus.query_once(goal)
    raw_moves = result.get("Moves", [])
    return [tuple(int(x) for x in m) for m in raw_moves]


# ── Move application ─────────────────────────────────────────────────────────

def apply_move(board: dict, move: tuple) -> dict:
    """
    Apply *move* = (from_col, from_row, to_col, to_row) to *board* and return
    the resulting new board dict.  Any piece previously on the destination
    square (a capture) is removed.  The original board is not mutated.
    """
    fc, fr, tc, tr = move
    new_board = dict(board)           # shallow copy is enough (values are immutable tuples)
    piece = new_board.pop((fc, fr))   # lift the moving piece
    new_board[(tc, tr)] = piece       # place it (overwrites captured piece if any)
    return new_board


# ── Win-condition query ───────────────────────────────────────────────────────

def check_winner(board: dict):
    """
    Query Prolog for a win condition on *board*.
    Returns "white", "black", "draw", or None.

    Kings-only guard: if both sides have zero pawns the Prolog win_condition
    would incorrectly declare White the winner (first matching clause fires).
    We intercept that case and return "draw" before calling Prolog.
    """
    has_white_pawn = any(v == ("white", "pawn") for v in board.values())
    has_black_pawn = any(v == ("black", "pawn") for v in board.values())
    if not has_white_pawn and not has_black_pawn:
        return "draw"
    board_str = board_to_prolog_str(board)
    result = janus.query_once(f"win_condition({board_str}, Winner)")
    if result:
        return result.get("Winner")
    return None


# ── FEN translation ──────────────────────────────────────────────────────────

# FEN piece symbols (board → FEN)
_FEN_SYMBOL = {
    ("white", "king"):  "K",
    ("white", "pawn"):  "P",
    ("black", "king"):  "k",
    ("black", "pawn"):  "p",
}

# FEN symbol → piece tuple (FEN → board).  Only kings and pawns are mapped;
# other symbols are silently ignored (pawn_wars.pl only knows kings and pawns).
_PIECE_FROM_FEN: dict[str, tuple] = {
    "K": ("white", "king"),
    "P": ("white", "pawn"),
    "k": ("black", "king"),
    "p": ("black", "pawn"),
}


def fen_to_board(fen: str) -> dict:
    """
    Parse a FEN string's piece-placement field into a board dict.
    Only kings and pawns are included; other pieces are skipped.
    The active-colour and remaining FEN fields are not parsed here —
    turn management is handled by _state in app.py.
    """
    placement = fen.split()[0]
    board: dict = {}
    row = 8
    col = 1
    for ch in placement:
        if ch == "/":
            row -= 1
            col = 1
        elif ch.isdigit():
            col += int(ch)
        else:
            piece = _PIECE_FROM_FEN.get(ch)
            if piece:
                board[(col, row)] = piece
            col += 1   # advance even for unrecognised symbols
    return board


def board_to_fen(board: dict) -> str:
    """
    Convert the custom board dict into a valid FEN string.
    This function is used EXCLUSIVELY for passing to python-chess's Board()
    constructor for SVG rendering.  It is never used for logic.

    Only piece placement and active-colour fields are meaningful here;
    the remaining FEN fields use placeholder values compatible with
    python-chess's parser.
    """
    rows = []
    for row in range(8, 0, -1):          # FEN ranks run 8 → 1
        empty = 0
        rank_str = ""
        for col in range(1, 9):
            piece = board.get((col, row))
            if piece is None:
                empty += 1
            else:
                if empty:
                    rank_str += str(empty)
                    empty = 0
                rank_str += _FEN_SYMBOL[piece]
        if empty:
            rank_str += str(empty)
        rows.append(rank_str)

    placement = "/".join(rows)
    # Active colour, castling, en-passant, halfmove, fullmove – all neutral
    return f"{placement} w - - 0 1"


# ── Algebraic notation helpers ────────────────────────────────────────────────

def alg_to_move(alg: str) -> tuple:
    """
    Parse a move string in the format "e2e4" into (from_col, from_row, to_col, to_row).
    Raises ValueError for malformed input.
    """
    alg = alg.strip().lower()
    if len(alg) != 4:
        raise ValueError(f"Move must be 4 characters (e.g. e2e4), got: {alg!r}")
    fc = letter_to_col(alg[0])
    fr = int(alg[1])
    tc = letter_to_col(alg[2])
    tr = int(alg[3])
    if not (1 <= fr <= 8 and 1 <= tr <= 8):
        raise ValueError(f"Row out of range in move: {alg!r}")
    return (fc, fr, tc, tr)


def move_to_alg(move: tuple) -> str:
    """Convert (from_col, from_row, to_col, to_row) to 'e2e4'-style string."""
    fc, fr, tc, tr = move
    return f"{col_to_letter(fc)}{fr}{col_to_letter(tc)}{tr}"
