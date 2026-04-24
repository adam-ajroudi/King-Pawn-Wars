"""
engine.py  –  Minimax with Alpha-Beta Pruning for Pawn Wars.

All game logic still lives in Prolog.  This module only contains:
  • a move cache to avoid redundant Prolog queries during recursion,
  • a static evaluation function (White's perspective),
  • Minimax with toggleable Alpha-Beta Pruning,
  • a node counter for comparative analysis (AB on vs. off),
  • a best_move() entry point called by the web server.
"""

import random

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
#
# Components:
#   1. Material         – 100 per pawn on each side.
#   2. Advancement      – accelerating bonus so pawns near promotion are prized
#                         far more than pawns just off the starting rank.
#   3. Passed pawn      – a pawn with no enemy pawn blocking its file or either
#                         adjacent file gets a large extra bonus.
#   4. King activity    – reward each king for being close to the nearest enemy
#                         pawn (to block or capture it) and to its own most
#                         advanced pawn (to escort it).

_PAWN_VALUE = 100

_ADVANCE_TABLE = [0, 10, 25, 50, 90, 150]

_PASSED_PAWN_BONUS = 60
_KING_BLOCK        = 4
_KING_ESCORT       = 2


def _chebyshev(c1: int, r1: int, c2: int, r2: int) -> int:
    """Chebyshev (king-move) distance between two squares."""
    return max(abs(c1 - c2), abs(r1 - r2))


def _is_passed(board: dict, color: str, col: int, row: int) -> bool:
    """
    True when the pawn of *color* at (col, row) has no opposing pawn on the
    same file or either adjacent file that is ahead of it.
    """
    opp = "black" if color == "white" else "white"
    for (pc, pr), (pcolor, ptype) in board.items():
        if pcolor != opp or ptype != "pawn":
            continue
        if abs(pc - col) <= 1:
            if color == "white" and pr > row:
                return False
            if color == "black" and pr < row:
                return False
    return True


def evaluate(board: dict) -> int:
    """
    Heuristic score from White's perspective.
    Higher = better for White; lower = better for Black.
    This function is White-positive throughout; color bias is
    handled in best_move() via the maximizing boolean.
    """
    white_pawns = [(c, r) for (c, r), (col, pt) in board.items()
                   if col == "white" and pt == "pawn"]
    black_pawns = [(c, r) for (c, r), (col, pt) in board.items()
                   if col == "black" and pt == "pawn"]
    white_king  = next(((c, r) for (c, r), (col, pt) in board.items()
                        if col == "white" and pt == "king"), None)
    black_king  = next(((c, r) for (c, r), (col, pt) in board.items()
                        if col == "black" and pt == "king"), None)

    score = 0

    for (col, row) in white_pawns:
        score += _PAWN_VALUE
        score += _ADVANCE_TABLE[min(row - 2, 5)]
        if _is_passed(board, "white", col, row):
            score += _PASSED_PAWN_BONUS

    for (col, row) in black_pawns:
        score -= _PAWN_VALUE
        score -= _ADVANCE_TABLE[min(7 - row, 5)]
        if _is_passed(board, "black", col, row):
            score -= _PASSED_PAWN_BONUS

    if white_king:
        wkc, wkr = white_king
        if black_pawns:
            d = min(_chebyshev(wkc, wkr, pc, pr) for pc, pr in black_pawns)
            score += (8 - d) * _KING_BLOCK
        if white_pawns:
            d = min(_chebyshev(wkc, wkr, pc, pr) for pc, pr in white_pawns)
            score += (8 - d) * _KING_ESCORT

    if black_king:
        bkc, bkr = black_king
        if white_pawns:
            d = min(_chebyshev(bkc, bkr, pc, pr) for pc, pr in white_pawns)
            score -= (8 - d) * _KING_BLOCK
        if black_pawns:
            d = min(_chebyshev(bkc, bkr, pc, pr) for pc, pr in black_pawns)
            score -= (8 - d) * _KING_ESCORT

    return score


# ── Minimax with Alpha-Beta Pruning ──────────────────────────────────────────

_INF = float("inf")

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
    maximizing     : True when it is White's turn to move
    node_counter   : single-element list [int] – incremented for every node visited
    use_alpha_beta : when False, pruning is disabled for comparative testing

    Returns
    -------
    float : minimax score from White's perspective
    """
    node_counter[0] += 1

    winner = check_winner(board)
    if winner == "white":
        return _WIN_SCORE + depth
    if winner == "black":
        return -(_WIN_SCORE + depth)

    if depth == 0:
        return evaluate(board)

    color = "white" if maximizing else "black"
    moves = cached_legal_moves(board, color)

    if not moves:
        return 0   # stalemate → draw

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
                    break
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
                    break
        return best


# ── Best move entry point ─────────────────────────────────────────────────────

def best_move(
    board: dict,
    depth: int = 3,
    use_alpha_beta: bool = True,
    ai_color: str = "black",
    disable_randomization: bool = False,
):
    """
    Run Minimax for the AI and return (chosen_move, node_count).

    Parameters
    ----------
    board                 : current board state
    depth                 : search depth (default 3)
    use_alpha_beta        : enable Alpha-Beta Pruning (default True)
    ai_color              : which color the AI is playing ("white" or "black")
    disable_randomization : when True, return the first best-scoring move
                            deterministically (used by compare endpoint to
                            guarantee both runs select the same move).
                            When False (default), randomly pick among tied moves
                            so the AI is less predictable in gameplay.

    Returns
    -------
    (move, node_count) where move is a (fc, fr, tc, tr) tuple or None.
    """
    moves = cached_legal_moves(board, ai_color)
    if not moves:
        return None, 0

    # maximizing=True when AI is White (evaluation is White-positive)
    maximizing_root = (ai_color == "white")
    best_val        = -_INF if maximizing_root else _INF
    best_moves: list = []
    node_counter    = [0]

    for move in moves:
        child_board = apply_move(board, move)
        val = minimax(
            child_board,
            depth - 1,
            -_INF,
            _INF,
            not maximizing_root,   # after AI's move, opponent's turn
            node_counter,
            use_alpha_beta,
        )
        if maximizing_root:
            if val > best_val:
                best_val   = val
                best_moves = [move]
            elif val == best_val:
                best_moves.append(move)
        else:
            if val < best_val:
                best_val   = val
                best_moves = [move]
            elif val == best_val:
                best_moves.append(move)

    if disable_randomization:
        chosen = best_moves[0]
    else:
        chosen = random.choice(best_moves)

    return chosen, node_counter[0]


# ── Comparative analysis helper ───────────────────────────────────────────────

def compare_node_counts(
    board: dict,
    depth: int = 3,
    ai_color: str = "black",
):
    """
    Run the search twice (with and without Alpha-Beta Pruning) and return a
    four-tuple: (count_ab, count_no_ab, move_ab, move_no_ab).

    Both runs use disable_randomization=True so move selection is deterministic
    and the two chosen moves can be compared for identity.
    Cache is cleared before each run to prevent cross-contamination.
    """
    clear_cache()
    move_ab, count_ab = best_move(
        board, depth,
        use_alpha_beta=True,
        ai_color=ai_color,
        disable_randomization=True,
    )
    clear_cache()
    move_no_ab, count_no_ab = best_move(
        board, depth,
        use_alpha_beta=False,
        ai_color=ai_color,
        disable_randomization=True,
    )

    reduction = (1 - count_ab / count_no_ab) * 100 if count_no_ab else 0
    print(f"Depth {depth} | AI={ai_color}")
    print(f"  With Alpha-Beta    : {count_ab:>8,} nodes  move={move_ab}")
    print(f"  Without Alpha-Beta : {count_no_ab:>8,} nodes  move={move_no_ab}")
    print(f"  Node reduction     : {reduction:.1f}%  identical={move_ab == move_no_ab}")

    return count_ab, count_no_ab, move_ab, move_no_ab
