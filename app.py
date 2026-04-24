"""
app.py  –  Pawn Wars web server.

Serves a single-page web UI backed by chessboard.js.
All game logic lives in Prolog (pawn_wars.pl).
Python handles the search engine and bridges to Prolog via Janus.

Endpoints
---------
GET  /                  – serve the game page
GET  /api/state         – current board state (FEN, legal moves, turn, winner, puzzle info)
POST /api/move          – apply a human move, run AI response
POST /api/new_game      – reset to a random (or tagged) puzzle position
POST /api/compare       – AB-on vs AB-off node comparison at independent depth
GET  /api/export        – download full move history as JSON
GET  /api/puzzles/tags  – sorted list of all available puzzle tags
"""

import json
import random as _random
import sys
from collections import defaultdict
from pathlib import Path

import janus_swi as janus
from flask import Flask, jsonify, render_template, request

from board import (
    apply_move, board_to_fen, check_winner, col_to_letter,
    fen_to_board, get_legal_moves, initial_board, alg_to_move, move_to_alg,
)
from engine import best_move, clear_cache, compare_node_counts, cached_legal_moves

app = Flask(__name__)


# ── Puzzle database ───────────────────────────────────────────────────────────
# Loaded once at startup.  All runtime puzzle selection reads from these.

_PUZZLES:   list[dict]        = []
_TAG_INDEX: dict[str, list[int]] = {}   # tag → list of indices into _PUZZLES


def _load_puzzles(path: str = "puzzles.json") -> None:
    """Load puzzles.json into module-level _PUZZLES and _TAG_INDEX."""
    global _PUZZLES, _TAG_INDEX
    p = Path(path)
    if not p.exists():
        print(f"[Puzzle DB] {path} not found – will use classic start position.")
        return
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    _PUZZLES = data
    for i, puzzle in enumerate(data):
        for tag in puzzle.get("tags", []):
            _TAG_INDEX.setdefault(tag, []).append(i)


# ── Server-side game state ────────────────────────────────────────────────────

_state: dict = {}


def _position_key(board: dict, turn: str) -> tuple:
    """Stable key for (board layout, side to move) — used for repetition draws."""
    return (tuple(sorted(board.items())), turn)


def _reset(depth: int = 4, puzzle: dict | None = None) -> None:
    """Reinitialise game state from a puzzle (or classic start if none available)."""
    clear_cache()

    if puzzle is None and _PUZZLES:
        puzzle = _random.choice(_PUZZLES)

    if puzzle is not None:
        b0              = fen_to_board(puzzle["fen"])
        human_color     = puzzle["human_color"]
        puzzle_fen      = puzzle["fen"]
        puzzle_solution = puzzle.get("solution", [])
        puzzle_sol_fens = puzzle.get("solution_fens", [])
        puzzle_id       = puzzle.get("id", "")
        puzzle_rating   = puzzle.get("rating", 0)
    else:
        # Fallback: classic Pawn Wars start (no puzzles.json)
        b0              = initial_board()
        human_color     = "white"
        puzzle_fen      = board_to_fen(b0)
        puzzle_solution = []
        puzzle_sol_fens = []
        puzzle_id       = ""
        puzzle_rating   = 0

    ai_color = "black" if human_color == "white" else "white"
    counts   = defaultdict(int)
    counts[_position_key(b0, human_color)] = 1

    _state.update(
        board              = b0,
        turn               = human_color,
        game_over          = False,
        winner             = None,
        last_ai_nodes      = 0,
        ai_depth           = depth,
        move_history       = [],
        fen_after_human    = None,
        position_counts    = counts,
        human_color        = human_color,
        ai_color           = ai_color,
        puzzle_id          = puzzle_id,
        puzzle_fen         = puzzle_fen,
        puzzle_solution    = puzzle_solution,
        puzzle_sol_fens    = puzzle_sol_fens,
        puzzle_rating      = puzzle_rating,
    )


def _king_in_pawn_check(board: dict, color: str) -> bool:
    """
    Return True when the king of *color* is on a square currently attacked
    by an enemy pawn.  Used to distinguish checkmate from stalemate when a
    player has zero legal moves.
    """
    king = next((sq for sq, v in board.items() if v == (color, "king")), None)
    if not king:
        return False
    kc, kr = king
    opp = "black" if color == "white" else "white"
    for (pc, pr), (c, t) in board.items():
        if c != opp or t != "pawn":
            continue
        if opp == "white" and pr + 1 == kr and abs(pc - kc) == 1:
            return True
        if opp == "black" and pr - 1 == kr and abs(pc - kc) == 1:
            return True
    return False


def _register_position(board: dict, turn: str) -> bool:
    """
    Record that (board, turn) has been reached again.
    Returns True if this is the third occurrence (draw by repetition).
    """
    key = _position_key(board, turn)
    _state["position_counts"][key] += 1
    return _state["position_counts"][key] >= 3


# ── Startup integrity checks ──────────────────────────────────────────────────

def _run_startup_checks() -> None:
    """
    Run four integrity checks at startup.  Prints PASS/FAIL for each.
    Exits with a specific error message if any check fails.
    """
    print("=" * 58)
    print("  Startup Integrity Checks")
    print("=" * 58)

    # ── Check 1: Prolog bridge ────────────────────────────────────────────────
    try:
        result = janus.query_once("opponent(white, X)")
        assert result.get("X") == "black", f"unexpected result: {result}"
        print("  [1] Prolog bridge              PASS")
    except Exception as exc:
        sys.exit(f"  [1] Prolog bridge              FAIL – {exc}")

    # ── Check 2: Puzzle database ──────────────────────────────────────────────
    count = len(_PUZZLES)
    if count == 0:
        print("  [2] Puzzle database            WARN – no puzzles loaded (classic start)")
    else:
        print(f"  [2] Puzzle database            PASS – {count:,} puzzles, "
              f"{len(_TAG_INDEX)} tags")

    # ── Check 3: Legal move assertion on known position ───────────────────────
    try:
        test_board = initial_board()
        moves      = get_legal_moves(test_board, "white")
        assert len(moves) > 0, "empty move list from initial position"
        # Verify at least one expected move is present (White King e1 can go to d1)
        assert (4, 1) in {(tc, tr) for (_, _, tc, tr) in moves}, \
            "expected king move e1->d1 not found"
        print(f"  [3] Legal move assertion       PASS – {len(moves)} moves from start")
    except Exception as exc:
        sys.exit(f"  [3] Legal move assertion       FAIL – {exc}")

    # ── Check 4: Transposition cache integrity ────────────────────────────────
    # Reach the same board position via two independent move sequences and
    # confirm cached_legal_moves returns identical results for both paths.
    try:
        from board import apply_move as _apply
        clear_cache()
        b0      = initial_board()
        wk_move = (5, 1, 4, 1)   # White King: e1 → d1
        bk_move = (5, 8, 4, 8)   # Black King: e8 → d8  (independent of WK move)

        # Path A: White King first, then Black King
        b_a = _apply(_apply(b0, wk_move), bk_move)
        # Path B: Black King first, then White King
        b_b = _apply(_apply(b0, bk_move), wk_move)

        assert b_a == b_b, "board positions differ – moves are not independent"

        moves_a = cached_legal_moves(b_a, "white")   # cache miss → Prolog query
        moves_b = cached_legal_moves(b_b, "white")   # cache hit  → same frozenset key

        assert sorted(moves_a) == sorted(moves_b), \
            "cache returned different moves for same position reached via different paths"
        print("  [4] Transposition cache        PASS")
    except Exception as exc:
        sys.exit(f"  [4] Transposition cache        FAIL – {exc}")

    print("=" * 58)
    print()


# ── Initialisation sequence ───────────────────────────────────────────────────

_load_puzzles()
_run_startup_checks()
_reset()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _legal_moves_js(board: dict, color: str) -> list:
    """
    Return legal moves as [{from:'e2', to:'e4'}, …] for the JS client.
    chessboard.js uses algebraic square names (e.g. 'e2', 'e4').
    """
    return [
        {
            "from": col_to_letter(m[0]) + str(m[1]),
            "to":   col_to_letter(m[2]) + str(m[3]),
        }
        for m in get_legal_moves(board, color)
    ]


def _state_payload() -> dict:
    """Build the JSON payload that describes the current game state."""
    board       = _state["board"]
    human_color = _state.get("human_color", "white")
    legal = (
        _legal_moves_js(board, human_color)
        if not _state["game_over"] and _state["turn"] == human_color
        else []
    )
    return {
        "fen":               board_to_fen(board),
        "fen_after_human":   _state.get("fen_after_human"),
        "turn":              _state["turn"],
        "game_over":         _state["game_over"],
        "winner":            _state["winner"],
        "legal_moves":       legal,
        "move_history":      _state["move_history"],
        "last_ai_nodes":     _state["last_ai_nodes"],
        "ai_depth":          _state["ai_depth"],
        "human_color":       human_color,
        "ai_color":          _state.get("ai_color", "black"),
        "puzzle_id":         _state.get("puzzle_id", ""),
        "puzzle_fen":        _state.get("puzzle_fen", ""),
        "puzzle_solution":   _state.get("puzzle_solution", []),
        "puzzle_sol_fens":   _state.get("puzzle_sol_fens", []),
        "puzzle_rating":     _state.get("puzzle_rating", 0),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(_state_payload())


@app.route("/api/puzzles/tags")
def api_puzzle_tags():
    """Return sorted list of all available puzzle tags."""
    # Report-specific tags always float to the top
    priority = ["passed_pawn_demo", "king_opposition_demo", "pawn_sacrifice_demo"]
    rest     = sorted(t for t in _TAG_INDEX if t not in priority)
    return jsonify(priority + rest)


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data  = request.get_json(silent=True) or {}
    depth = max(1, min(6, int(data.get("depth", _state["ai_depth"]))))
    tag   = (data.get("tag") or "").strip()

    puzzle = None
    if tag and tag in _TAG_INDEX:
        puzzle = _PUZZLES[_random.choice(_TAG_INDEX[tag])]
    elif _PUZZLES:
        puzzle = _random.choice(_PUZZLES)

    _reset(depth, puzzle)
    return jsonify({"status": "ok", **_state_payload()})


@app.route("/api/replay", methods=["POST"])
def api_replay():
    """Restart the current puzzle from its original starting position."""
    data  = request.get_json(silent=True) or {}
    depth = max(1, min(6, int(data.get("depth", _state["ai_depth"]))))

    # Try to find the exact same puzzle by ID first
    puzzle_id = _state.get("puzzle_id", "")
    puzzle = next((p for p in _PUZZLES if p.get("id") == puzzle_id), None)

    # Fallback: rebuild a minimal puzzle dict from the stored state
    if puzzle is None and _state.get("puzzle_fen"):
        puzzle = {
            "fen":           _state["puzzle_fen"],
            "human_color":   _state.get("human_color", "white"),
            "solution":      _state.get("puzzle_solution", []),
            "solution_fens": _state.get("puzzle_sol_fens", []),
            "id":            _state.get("puzzle_id", ""),
            "rating":        _state.get("puzzle_rating", 0),
        }

    _reset(depth, puzzle)
    return jsonify({"status": "ok", **_state_payload()})


@app.route("/api/move", methods=["POST"])
def api_move():
    if _state["game_over"]:
        return jsonify({"status": "error", "message": "Game is already over."}), 400

    human_color = _state["human_color"]
    ai_color    = _state["ai_color"]

    if _state["turn"] != human_color:
        return jsonify({"status": "error", "message": "Not your turn."}), 400

    data    = request.get_json()
    from_sq = data.get("from", "")
    to_sq   = data.get("to",   "")
    alg     = from_sq + to_sq

    board = _state["board"]
    legal = get_legal_moves(board, human_color)

    try:
        chosen = alg_to_move(alg)
    except ValueError as exc:
        return jsonify({"status": "illegal", "message": str(exc)}), 400

    if chosen not in legal:
        return jsonify({"status": "illegal",
                        "message": f"Illegal move: {alg}"}), 400

    # Snapshot FEN before human's move (used in history export)
    fen_before_human = board_to_fen(board)

    # ── Apply human move ──────────────────────────────────────────────────────
    board = apply_move(board, chosen)
    _state["board"]          = board
    _state["fen_after_human"] = board_to_fen(board)

    # Build history entry with chess-color keys (white/black)
    history_entry: dict = {
        "white":               alg if human_color == "white" else None,
        "black":               alg if human_color == "black" else None,
        "nodes":               None,
        "fen_before":          fen_before_human,
        "alpha_beta_enabled":  True,
    }

    winner = check_winner(board)
    if winner:
        _state.update(game_over=True, winner=winner, turn="none")
        if human_color == "white":
            history_entry["black"] = "—"
        else:
            history_entry["white"] = "—"
        _state["move_history"].append(history_entry)
        return jsonify({"status": "ok", "ai_move": None, **_state_payload()})

    if _register_position(board, ai_color):
        _state.update(game_over=True, winner="draw_repetition", turn="none")
        if human_color == "white":
            history_entry["black"] = "(draw)"
        else:
            history_entry["white"] = "(draw)"
        _state["move_history"].append(history_entry)
        return jsonify({"status": "ok", "ai_move": None, **_state_payload()})

    # ── AI responds ───────────────────────────────────────────────────────────
    _state["turn"] = ai_color
    ai_move_tuple, nodes = best_move(
        board,
        depth=_state["ai_depth"],
        ai_color=ai_color,
        disable_randomization=False,
    )
    _state["last_ai_nodes"] = nodes

    if ai_move_tuple is None:
        # Distinguish pawn-checkmate from stalemate:
        # if AI king is under pawn attack → attacker (human) wins; else draw.
        ai_in_check = _king_in_pawn_check(board, ai_color)
        if ai_in_check:
            _state.update(game_over=True, winner=human_color, turn="none")
        else:
            _state.update(game_over=True, winner=None, turn="none")
        no_move_label = "checkmate" if ai_in_check else "stalemate"
        if ai_color == "black":
            history_entry["black"] = no_move_label
        else:
            history_entry["white"] = no_move_label
        _state["move_history"].append(history_entry)
        return jsonify({"status": "ok", "ai_move": None, **_state_payload()})

    board     = apply_move(board, ai_move_tuple)
    _state["board"] = board
    ai_alg    = move_to_alg(ai_move_tuple)

    if ai_color == "black":
        history_entry["black"] = ai_alg
    else:
        history_entry["white"] = ai_alg
    history_entry["nodes"] = nodes

    winner = check_winner(board)
    if winner:
        _state.update(game_over=True, winner=winner, turn="none")
    elif _register_position(board, human_color):
        _state.update(game_over=True, winner="draw_repetition", turn="none")
    else:
        _state["turn"] = human_color
        # Check if human now has no legal moves (stalemate / checkmate)
        human_moves = get_legal_moves(board, human_color)
        if not human_moves:
            human_in_check = _king_in_pawn_check(board, human_color)
            if human_in_check:
                _state.update(game_over=True, winner=ai_color, turn="none")
            else:
                _state.update(game_over=True, winner=None, turn="none")

    _state["move_history"].append(history_entry)
    return jsonify({"status": "ok", "ai_move": ai_alg, **_state_payload()})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Run Minimax from the current position twice (AB on, AB off) at the
    independently selected compare depth.  Both runs use disable_randomization=True
    so chosen moves are deterministic and can be compared for identity.
    """
    data          = request.get_json(silent=True) or {}
    compare_depth = max(2, min(5, int(data.get("compare_depth", 3))))
    board         = _state["board"]

    # Always analyze from the AI's color perspective — the comparison is about
    # measuring engine efficiency (AB vs plain Minimax), not about the human's moves.
    analyze_color = _state.get("ai_color", "black")

    count_ab, count_no_ab, move_ab, move_no_ab = compare_node_counts(
        board, depth=compare_depth, ai_color=analyze_color
    )
    reduction      = (1 - count_ab / count_no_ab) * 100 if count_no_ab else 0
    move_ab_alg    = move_to_alg(move_ab)    if move_ab    else "—"
    move_no_ab_alg = move_to_alg(move_no_ab) if move_no_ab else "—"

    return jsonify({
        "depth":           compare_depth,
        "analyze_color":   analyze_color,
        "with_ab":         count_ab,
        "without_ab":      count_no_ab,
        "reduction":       round(reduction, 1),
        "move_ab":         move_ab_alg,
        "move_noab":       move_no_ab_alg,
        "moves_identical": move_ab == move_no_ab,
    })


@app.route("/api/export")
def api_export():
    """Return the current game's move history as a JSON array for download."""
    history = _state.get("move_history", [])
    records = [
        {
            "move_number":        i + 1,
            "white":              entry.get("white") or "—",
            "black":              entry.get("black") or "—",
            "fen":                entry.get("fen_before", ""),
            "ai_nodes":           entry.get("nodes"),
            "alpha_beta_enabled": entry.get("alpha_beta_enabled", True),
        }
        for i, entry in enumerate(history)
    ]
    return jsonify(records)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("  Pawn Wars  –  http://localhost:5000")
    print("=" * 58)
    app.run(host="127.0.0.1", port=5000, debug=False)
