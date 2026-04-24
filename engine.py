"""
engine.py  –  Minimax with Alpha-Beta Pruning for Pawn Wars.

All game logic still lives in Prolog.  This module only contains:
  • a move cache to avoid redundant Prolog queries during recursion,
  • a static evaluation function (White's perspective),
  • Minimax with toggleable Alpha-Beta Pruning,
  • a node counter for comparative analysis (AB on vs. off),
  • a best_move() entry point called by the notebook game loop.
"""

from board import get_legal_moves, apply_move, check_winner

# ── Move cache ────────────────────────────────────────────────────────────────
# Keyed by (board_frozenset, color).  Stores the list of legal move tuples
# so that repeated Prolog queries for the same state are avoided during search.

_move_cache: dict = {}


def _board_key(board: dict) -> frozenset:
    """Convert the board dict to a hashable frozenset for cache keying."""
    return frozenset(board.items())


def cached_legal_moves(board: dict, color: str) -> list:
    """
    Return legal moves for *color* on *board*, consulting the cache first.
    Queries Prolog only on a cache miss, then stores the result.
    This prevents bridge performance collapse during deep recursion.
    """
    key = (_board_key(board), color)
    if key not in _move_cache:
        _move_cache[key] = get_legal_moves(board, color)
    return _move_cache[key]


def clear_cache() -> None:
    """Clear the move cache between games or experiments."""
    _move_cache.clear()


# ── Static evaluation function ────────────────────────────────────────────────
# Scores the board from White's perspective.
# Positive values favour White; negative values favour Black.

_PAWN_VALUE = 100       # material value of each pawn
_ADVANCE_BONUS = 12     # bonus per rank of pawn advancement beyond starting rank
_KING_VALUE = 0         # kings are always present; no differential value added


def evaluate(board: dict) -> int:
    """
    Heuristic score from White's perspective:
      +100 per white pawn   – 100 per black pawn   (material)
      +12  per rank a white pawn has advanced beyond row 2
      –12  per rank a black pawn has advanced beyond row 7 (i.e. descending)
    """
    score = 0
    for (col, row), (color, ptype) in board.items():
        if ptype == "pawn":
            if color == "white":
                score += _PAWN_VALUE
                score += (row - 2) * _ADVANCE_BONUS   # row 2 → 0, row 7 → +60
            else:
                score -= _PAWN_VALUE
                score -= (7 - row) * _ADVANCE_BONUS   # row 7 → 0, row 2 → -60
    return score


# ── Minimax with Alpha-Beta Pruning ──────────────────────────────────────────

_INF = float("inf")

# Terminal scores are offset by remaining depth so the engine prefers
# faster wins (more depth remaining when terminal is found = found sooner).
_WIN_SCORE = 100_000


def minimax(
    board: dict,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    node_counter: list,
    use_alpha_beta: bool = True,
) -> float:
    """
    Minimax search with optional Alpha-Beta Pruning.

    Parameters
    ----------
    board          : current board state (dict)
    depth          : remaining search depth (0 → evaluate leaf)
    alpha          : best score the maximiser can guarantee so far
    beta           : best score the minimiser can guarantee so far
    maximizing     : True when it is White's turn
    node_counter   : single-element list [int] – incremented for every node
                     visited (pass by reference via mutable list)
    use_alpha_beta : when False, pruning is disabled for comparative testing

    Returns
    -------
    float : minimax score from White's perspective
    """
    node_counter[0] += 1

    # ── Terminal-state detection ──────────────────────────────────────────────
    winner = check_winner(board)
    if winner == "white":
        # Reward finding the win sooner (higher depth remaining = earlier win)
        return _WIN_SCORE + depth
    if winner == "black":
        return -(_WIN_SCORE + depth)

    # ── Leaf node ─────────────────────────────────────────────────────────────
    if depth == 0:
        return evaluate(board)

    # ── Recursive search ──────────────────────────────────────────────────────
    color = "white" if maximizing else "black"
    moves = cached_legal_moves(board, color)

    # A side with no legal moves but no win condition is stalemated → draw (0)
    if not moves:
        return 0

    if maximizing:
        best = -_INF
        for move in moves:
            child_board = apply_move(board, move)
            val = minimax(child_board, depth - 1, alpha, beta, False,
                          node_counter, use_alpha_beta)
            best = max(best, val)
            if use_alpha_beta:
                alpha = max(alpha, best)
                if beta <= alpha:
                    break   # β-cutoff
        return best
    else:
        best = _INF
        for move in moves:
            child_board = apply_move(board, move)
            val = minimax(child_board, depth - 1, alpha, beta, True,
                          node_counter, use_alpha_beta)
            best = min(best, val)
            if use_alpha_beta:
                beta = min(beta, best)
                if beta <= alpha:
                    break   # α-cutoff
        return best


# ── Best move entry point ─────────────────────────────────────────────────────

def best_move(board: dict, depth: int = 3, use_alpha_beta: bool = True):
    """
    Run Minimax for Black at the given *depth* and return the chosen move and
    the total node count for the search.

    Parameters
    ----------
    board          : current board state
    depth          : search depth (default 3)
    use_alpha_beta : enable Alpha-Beta Pruning (default True)

    Returns
    -------
    (move, node_count) where move is a (fc, fr, tc, tr) tuple or None if
    Black has no legal moves.
    """
    moves = cached_legal_moves(board, "black")
    if not moves:
        return None, 0

    node_counter = [0]
    best_val = _INF
    chosen_move = None

    for move in moves:
        child_board = apply_move(board, move)
        val = minimax(
            child_board,
            depth - 1,
            -_INF,
            _INF,
            True,          # after Black moves, it is White's turn
            node_counter,
            use_alpha_beta,
        )
        if val < best_val:
            best_val = val
            chosen_move = move

    return chosen_move, node_counter[0]


# ── Comparative analysis helper ───────────────────────────────────────────────

def compare_node_counts(board: dict, depth: int = 3):
    """
    Run the search twice (with and without Alpha-Beta Pruning) and print a
    side-by-side comparison of node counts.  Useful for the report section.
    """
    clear_cache()
    move_ab, count_ab = best_move(board, depth, use_alpha_beta=True)
    clear_cache()
    move_no_ab, count_no_ab = best_move(board, depth, use_alpha_beta=False)

    print(f"Depth {depth} search node counts")
    print(f"  With Alpha-Beta Pruning : {count_ab:>8,} nodes  -> move {move_ab}")
    print(f"  Without Alpha-Beta      : {count_no_ab:>8,} nodes  -> move {move_no_ab}")
    reduction = (1 - count_ab / count_no_ab) * 100 if count_no_ab else 0
    print(f"  Node reduction          : {reduction:.1f}%")
    return count_ab, count_no_ab
