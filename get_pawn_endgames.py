"""
Extract Lichess pawn-endgame puzzles from CSV into PGN.

Usage:
    python get_pawn_endgames.py
    python get_pawn_endgames.py --input lichess_db_puzzle.csv --output lichess_pawn_endgames.pgn
    python get_pawn_endgames.py --theme pawnEndgame
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Lichess puzzle CSV by theme and export PGN."
    )
    parser.add_argument(
        "--input",
        default="lichess_db_puzzle.csv",
        help="Path to input Lichess puzzle CSV file.",
    )
    parser.add_argument(
        "--output",
        default="lichess_pawn_endgames.pgn",
        help="Path to output PGN file.",
    )
    parser.add_argument(
        "--theme",
        default="pawnEndgame",
        help="Theme token to filter by (default: pawnEndgame).",
    )
    return parser.parse_args()


def build_pgn_block(row: dict[str, str]) -> str:
    return (
        f'[Event "Lichess Puzzle {row["PuzzleId"]}"]\n'
        f'[Site "{row["GameUrl"]}"]\n'
        f'[FEN "{row["FEN"]}"]\n'
        '[SetUp "1"]\n'
        f'[Rating "{row["Rating"]}"]\n'
        f'\n{row["Moves"]}\n\n'
    )


def extract_theme_to_pgn(input_path: Path, output_path: Path, theme: str) -> int:
    count = 0

    with input_path.open("r", newline="", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        required_columns = {
            "PuzzleId",
            "FEN",
            "Moves",
            "Rating",
            "Themes",
            "GameUrl",
        }

        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns.difference(set(reader.fieldnames or [])))
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )

        for row in reader:
            themes = row.get("Themes", "")
            if theme in themes.split():
                f_out.write(build_pgn_block(row))
                count += 1

    return count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Processing {input_path} ...")
    extracted = extract_theme_to_pgn(input_path, output_path, args.theme)
    print(f"Done. Extracted {extracted} puzzles to {output_path}")


if __name__ == "__main__":
    main()
