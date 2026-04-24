"""
build_puzzle_db.py – Preprocess Lichess pawn-endgame puzzles into a runtime JSON database.

Reads lichess_db_puzzle.csv, filters for pawnEndgame theme, applies moves_uci[0]
to the raw FEN (the opponent's setup move that frames the puzzle), stores the resulting
FEN as the play position, derives human_color from that FEN's active color, and
pre-computes the full FEN sequence for solution replay animation.

python-chess is used here for move application.
This script is an OFFLINE preprocessing utility – it does not run during the game
and has no bearing on the project's academic constraints.

Output: puzzles.json  (used at runtime; the PGN is never read by the app directly)

Usage:
    python build_puzzle_db.py
    python build_puzzle_db.py --input lichess_db_puzzle.csv --output puzzles.json --limit 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import chess  # python-chess: ALLOWED only in this offline preprocessing script

# ── Demo tag assignments ─────────────────────────────────────────────────────
# The FIRST puzzle that qualifies for each theme gets an extra named tag so the
# three report-specific positions can be loaded by name from the UI.
_DEMO_SLOTS: list[tuple[str, str]] = [
    ("advancedPawn", "passed_pawn_demo"),     # closest to passed-pawn structure in pawnEndgame
    ("zugzwang",     "king_opposition_demo"),  # zugzwang = king opposition / key square
    ("promotion",    "pawn_sacrifice_demo"),   # pawn race / sacrifice to promote
]

_KINGS_AND_PAWNS = {chess.KING, chess.PAWN}


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build puzzles.json from lichess_db_puzzle.csv."
    )
    p.add_argument(
        "--input", default="lichess_db_puzzle.csv",
        help="Path to Lichess puzzle CSV (default: lichess_db_puzzle.csv).",
    )
    p.add_argument(
        "--output", default="puzzles.json",
        help="Output JSON file (default: puzzles.json).",
    )
    p.add_argument(
        "--limit", type=int, default=5000,
        help="Max puzzles to include, 0 = no limit (default: 5000).",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _only_kings_and_pawns(board: chess.Board) -> bool:
    return all(p.piece_type in _KINGS_AND_PAWNS for p in board.piece_map().values())


def _build_solution_fens(start_fen: str, solution_uci: list[str]) -> list[str]:
    """
    Pre-compute the FEN after each solution move for JS replay animation.
    Returns [start_fen, fen_after_move1, fen_after_move2, ...].
    Stops early if a move is illegal in the position.
    """
    fens = [start_fen]
    b = chess.Board(start_fen)
    for uci in solution_uci:
        move = chess.Move.from_uci(uci)
        if move not in b.legal_moves:
            break
        b.push(move)
        fens.append(b.fen())
    return fens


# ── Main build routine ───────────────────────────────────────────────────────

def build_db(input_path: Path, output_path: Path, limit: int) -> None:
    demo_pending: dict[str, str] = dict(_DEMO_SLOTS)   # theme_key → demo_tag
    demo_done:    set[str]       = set()
    records:      list[dict]     = []
    skipped = 0
    seen    = 0

    print(f"Reading {input_path} ...")

    with input_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            seen += 1
            if seen % 500_000 == 0:
                print(f"  … scanned {seen:,} rows, kept {len(records):,}")
            if limit and len(records) >= limit:
                break

            themes_str = row.get("Themes", "")
            themes     = themes_str.split()
            if "pawnEndgame" not in themes:
                continue

            moves_raw = row.get("Moves", "").strip()
            moves_uci = moves_raw.split() if moves_raw else []
            if len(moves_uci) < 2:
                skipped += 1
                continue

            raw_fen = row.get("FEN", "")
            try:
                setup_board = chess.Board(raw_fen)
                setup_move  = chess.Move.from_uci(moves_uci[0])
                if setup_move not in setup_board.legal_moves:
                    skipped += 1
                    continue
                setup_board.push(setup_move)

                # Skip positions that contain non-pawn, non-king pieces
                # (pawn_wars.pl only understands kings and pawns)
                if not _only_kings_and_pawns(setup_board):
                    skipped += 1
                    continue

                play_fen    = setup_board.fen()
                human_color = "white" if setup_board.turn == chess.WHITE else "black"
                solution    = moves_uci[1:]

                solution_fens = _build_solution_fens(play_fen, solution)

            except Exception:
                skipped += 1
                continue

            # Assign demo tags to the first qualifying puzzle for each theme
            extra_tags: list[str] = []
            for theme_key, demo_tag in list(demo_pending.items()):
                if theme_key in themes and demo_tag not in demo_done:
                    extra_tags.append(demo_tag)
                    demo_done.add(demo_tag)
                    del demo_pending[theme_key]

            try:
                rating = int(row["Rating"])
            except (ValueError, KeyError):
                rating = 0

            records.append({
                "id":             row["PuzzleId"],
                "fen":            play_fen,
                "human_color":    human_color,
                "solution":       solution,
                "solution_fens":  solution_fens,
                "rating":         rating,
                "tags":           themes + extra_tags,
            })

    print(f"Writing {len(records):,} puzzles ({skipped:,} skipped) -> {output_path} ...")
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, separators=(",", ":"))

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"Done. Output: {size_mb:.1f} MB")

    print("\nDemo tag assignments:")
    for _, demo_tag in _DEMO_SLOTS:
        status = "assigned" if demo_tag in demo_done else "NOT FOUND – raise --limit or adjust theme filter"
        print(f"  {demo_tag}: {status}")


def main() -> None:
    args  = parse_args()
    in_p  = Path(args.input)
    out_p = Path(args.output)

    if not in_p.exists():
        sys.exit(f"Input file not found: {in_p}")

    build_db(in_p, out_p, args.limit)


if __name__ == "__main__":
    main()
