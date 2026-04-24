"""
app.py  –  Pawn Wars web server.

Serves a single-page web UI backed by chessboard.js.
All game logic lives in Prolog (pawn_wars.pl).
Python handles the search engine and bridges to Prolog via Janus.

Endpoints
---------
GET  /               – serve the game page
GET  /api/state      – current board state (FEN, legal moves, turn, winner)
POST /api/move       – apply a human move, run AI response
POST /api/new_game   – reset to initial position
POST /api/compare    – node-count comparison (AB on vs off) from current position
"""

from collections import defaultdict

from flask import Flask, render_template, jsonify, request

from board import (
    initial_board, get_legal_moves, apply_move,
    check_winner, board_to_fen, alg_to_move, move_to_alg, col_to_letter,
)
from engine import best_move, compare_node_counts, clear_cache

app = Flask(__name__)


# ── Server-side game state ────────────────────────────────────────────────────
# A plain dict is sufficient for a local single-player server.

_state: dict = {}


def _position_key(board: dict, turn: str) -> tuple:
    """Stable key for (board layout, side to move) — used for repetition draws."""
    return (tuple(sorted(board.items())), turn)


def _reset(depth: int = 4) -> None:
    """Reinitialise game state to the starting position."""
    clear_cache()
    b0 = initial_board()
    counts = defaultdict(int)
    counts[_position_key(b0, "white")] = 1
    _state.update(
        board         = b0,
        turn          = "white",
        game_over     = False,
        winner        = None,
        last_ai_nodes = 0,
        ai_depth      = depth,
        move_history  = [],        # list of {white, black, nodes}
        fen_after_white = None,    # FEN snapshot sent to JS for animation
        position_counts = counts,  # threefold-style: same key 3 times → draw
    )


def _register_position(board: dict, turn: str) -> bool:
    """
    Record that (board, turn) has been reached again.
    Returns True if this is the third occurrence (draw by repetition).
    """
    key = _position_key(board, turn)
    _state["position_counts"][key] += 1
    return _state["position_counts"][key] >= 3


_reset()   # initialise once at startup


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
    board = _state["board"]
    legal = (
        _legal_moves_js(board, "white")
        if not _state["game_over"] and _state["turn"] == "white"
        else []
    )
    return {
        "fen":              board_to_fen(board),
        "fen_after_white":  _state.get("fen_after_white"),
        "turn":             _state["turn"],
        "game_over":        _state["game_over"],
        "winner":           _state["winner"],
        "legal_moves":      legal,
        "move_history":     _state["move_history"],
        "last_ai_nodes":    _state["last_ai_nodes"],
        "ai_depth":         _state["ai_depth"],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(_state_payload())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(silent=True) or {}
    depth = max(1, min(6, int(data.get("depth", _state["ai_depth"]))))
    _reset(depth)
    return jsonify({"status": "ok", **_state_payload()})


@app.route("/api/move", methods=["POST"])
def api_move():
    if _state["game_over"]:
        return jsonify({"status": "error", "message": "Game is already over."}), 400
    if _state["turn"] != "white":
        return jsonify({"status": "error", "message": "Not your turn."}), 400

    data     = request.get_json()
    from_sq  = data.get("from", "")   # e.g. "e2"
    to_sq    = data.get("to",   "")   # e.g. "e4"
    alg      = from_sq + to_sq        # "e2e4"

    # ── Validate against Prolog-generated legal moves ─────────────────────────
    board = _state["board"]
    legal = get_legal_moves(board, "white")

    try:
        chosen = alg_to_move(alg)
    except ValueError as exc:
        return jsonify({"status": "illegal", "message": str(exc)}), 400

    if chosen not in legal:
        return jsonify({"status": "illegal",
                        "message": f"Illegal move: {alg}"}), 400

    # ── Apply human move ──────────────────────────────────────────────────────
    board = apply_move(board, chosen)
    _state["board"] = board
    _state["fen_after_white"] = board_to_fen(board)   # snapshot for animation
    history_entry = {"white": alg, "black": None, "nodes": None}

    # Check if White just won (pawn promotion / last black pawn captured)
    winner = check_winner(board)
    if winner:
        _state.update(game_over=True, winner=winner, turn="none")
        history_entry["black"] = "—"
        _state["move_history"].append(history_entry)
        return jsonify({"status": "ok", "ai_move": None, **_state_payload()})

    if _register_position(board, "black"):
        _state.update(game_over=True, winner="draw", turn="none")
        history_entry["black"] = "(draw)"
        _state["move_history"].append(history_entry)
        return jsonify({"status": "ok", "ai_move": None, **_state_payload()})

    # ── AI (Black) responds ───────────────────────────────────────────────────
    _state["turn"] = "black"
    ai_move, nodes = best_move(board, depth=_state["ai_depth"])
    _state["last_ai_nodes"] = nodes

    if ai_move is None:
        # Black has no legal moves → stalemate
        _state.update(game_over=True, turn="none")
        history_entry["black"] = "stalemate"
        _state["move_history"].append(history_entry)
        return jsonify({"status": "ok", "ai_move": None, **_state_payload()})

    board = apply_move(board, ai_move)
    _state["board"] = board
    ai_alg = move_to_alg(ai_move)
    history_entry.update(black=ai_alg, nodes=nodes)

    # Check if Black just won
    winner = check_winner(board)
    if winner:
        _state.update(game_over=True, winner=winner, turn="none")
    elif _register_position(board, "white"):
        _state.update(game_over=True, winner="draw", turn="none")
    else:
        _state["turn"] = "white"

    _state["move_history"].append(history_entry)
    return jsonify({"status": "ok", "ai_move": ai_alg, **_state_payload()})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Run Minimax from the current position twice – once with Alpha-Beta Pruning
    and once without – and return the node counts for both runs.
    Used by the sidebar comparison panel for the report.
    """
    data  = request.get_json(silent=True) or {}
    depth = int(data.get("depth", _state["ai_depth"]))
    board = _state["board"]

    count_ab, count_no_ab = compare_node_counts(board, depth=depth)
    reduction = (1 - count_ab / count_no_ab) * 100 if count_no_ab else 0

    return jsonify({
        "depth":      depth,
        "with_ab":    count_ab,
        "without_ab": count_no_ab,
        "reduction":  round(reduction, 1),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Pawn Wars  –  http://localhost:5000")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=False)
