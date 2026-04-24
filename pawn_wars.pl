%% pawn_wars.pl
%% Pawn Wars game logic knowledge base.
%% All move generation, rule validation, and win conditions live here.
%% King moves filter destinations using direct board checks (enemy pawn attacks,
%% enemy king adjacency) so the rule engine stays non-circular.
%%
%% Board representation:
%%   A board is a list of piece/4 terms: piece(Color, Type, Col, Row)
%%   Color : white | black
%%   Type  : pawn  | king
%%   Col   : 1–8  (a=1, b=2, ..., h=8)
%%   Row   : 1–8


%% ─── Colour helpers ────────────────────────────────────────────────────────

%% opponent/2 – relates each colour to its adversary.
opponent(white, black).
opponent(black, white).


%% ─── Square occupancy helpers ───────────────────────────────────────────────

%% occupied/3 – true when any piece occupies (Col, Row) on Board.
occupied(Board, Col, Row) :-
    member(piece(_, _, Col, Row), Board).

%% friendly/4 – true when a piece of Color occupies (Col, Row).
friendly(Board, Color, Col, Row) :-
    member(piece(Color, _, Col, Row), Board).

%% enemy/4 – true when an opponent's piece occupies (Col, Row).
enemy(Board, Color, Col, Row) :-
    opponent(Color, Opp),
    member(piece(Opp, _, Col, Row), Board).


%% ─── White pawn moves ───────────────────────────────────────────────────────

%% white_pawn_moves/4 – collects all legal moves for a white pawn at (Col, Row).
%% White pawns advance toward increasing row numbers.
white_pawn_moves(Board, Col, Row, Moves) :-
    Row1 is Row + 1,

    %% Single-square advance: destination must be empty.
    ( Row1 =< 8, \+ occupied(Board, Col, Row1)
    -> Advance = [move(Col, Row, Col, Row1)]
    ;  Advance = [] ),

    %% Two-square push: only from starting rank (row 2), both squares must be empty.
    Row2 is Row + 2,
    ( Row =:= 2,
      \+ occupied(Board, Col, Row1),
      \+ occupied(Board, Col, Row2)
    -> Double = [move(Col, Row, Col, Row2)]
    ;  Double = [] ),

    %% Diagonal captures: one step diagonally forward onto an enemy PAWN only.
    %% Kings are not capturable – they must never be removed from the board.
    findall(
        move(Col, Row, CaptCol, Row1),
        ( member(DCol, [-1, 1]),
          CaptCol is Col + DCol,
          CaptCol >= 1, CaptCol =< 8,
          Row1 =< 8,
          member(piece(black, pawn, CaptCol, Row1), Board) ),
        Captures
    ),

    append([Advance, Double, Captures], Moves).


%% ─── Black pawn moves ───────────────────────────────────────────────────────

%% black_pawn_moves/4 – collects all legal moves for a black pawn at (Col, Row).
%% Black pawns advance toward decreasing row numbers (mirrored from white).
black_pawn_moves(Board, Col, Row, Moves) :-
    Row1 is Row - 1,

    %% Single-square advance: destination must be empty.
    ( Row1 >= 1, \+ occupied(Board, Col, Row1)
    -> Advance = [move(Col, Row, Col, Row1)]
    ;  Advance = [] ),

    %% Two-square push: only from starting rank (row 7), both squares must be empty.
    Row2 is Row - 2,
    ( Row =:= 7,
      \+ occupied(Board, Col, Row1),
      \+ occupied(Board, Col, Row2)
    -> Double = [move(Col, Row, Col, Row2)]
    ;  Double = [] ),

    %% Diagonal captures: one step diagonally forward (downward) onto an enemy PAWN only.
    %% Kings are not capturable – they must never be removed from the board.
    findall(
        move(Col, Row, CaptCol, Row1),
        ( member(DCol, [-1, 1]),
          CaptCol is Col + DCol,
          CaptCol >= 1, CaptCol =< 8,
          Row1 >= 1,
          member(piece(white, pawn, CaptCol, Row1), Board) ),
        Captures
    ),

    append([Advance, Double, Captures], Moves).


%% ─── Pawn attack detection (no move generation — no circular dependency) ────

%% pawn_attacks_square/5 – true when a pawn of PawnColor sitting at (PC, PR)
%% attacks the square (TC, TR).
%% White pawns attack diagonally upward   (TR = PR + 1).
%% Black pawns attack diagonally downward (TR = PR - 1).
%% Uses only arithmetic checks; never calls the move generator.
pawn_attacks_square(white, PC, PR, TC, TR) :-
    TR =:= PR + 1,
    ( TC =:= PC - 1 ; TC =:= PC + 1 ).

pawn_attacks_square(black, PC, PR, TC, TR) :-
    TR =:= PR - 1,
    ( TC =:= PC - 1 ; TC =:= PC + 1 ).

%% square_safe_for_king/4 – true when (TC, TR) is not attacked by any enemy pawn.
%% Implemented via direct board inspection rather than the move generator,
%% which avoids the circular dependency described in the original specification.
square_safe_for_king(Board, Color, TC, TR) :-
    opponent(Color, Opp),
    \+ (
        member(piece(Opp, pawn, PC, PR), Board),
        pawn_attacks_square(Opp, PC, PR, TC, TR)
    ).

%% king_square_avoids_enemy_king/4 – true when (ToCol, ToRow) lies outside the
%% opponent king’s Chebyshev-1 zone (same rule as “king may not move into check”
%% from the other king: the two kings may never occupy adjacent squares).
king_square_avoids_enemy_king(Board, Color, ToCol, ToRow) :-
    opponent(Color, Opp),
    member(piece(Opp, king, EKC, EKR), Board),
    DC is abs(ToCol - EKC),
    DR is abs(ToRow - EKR),
    \+ (DC =< 1, DR =< 1).


%% ─── King moves ─────────────────────────────────────────────────────────────

%% king_moves/5 – collects all legal moves for a king of Color at (Col, Row).
%% The king may step one square in any of the eight directions, provided:
%%   • the destination is within the 8×8 board,
%%   • the destination is not occupied by a friendly piece,
%%   • the destination is not occupied by any king (kings are never capturable),
%%   • the destination is not adjacent to the enemy king (kings may not touch),
%%   • the destination is not attacked by an enemy pawn.
%%     (Pawn attack and king-distance checks use direct board inspection rather
%%     than the move generator, so there is no circular dependency.)
%%
%% NOTE: \+ (DC =:= 0, DR =:= 0) is used instead of (DC \= 0 ; DR \= 0)
%% because the disjunction form is non-deterministic inside findall: when both
%% branches succeed (e.g. DC=1, DR=1) the same move is collected twice.
king_moves(Board, Color, Col, Row, Moves) :-
    findall(
        move(Col, Row, ToCol, ToRow),
        ( member(DC, [-1, 0, 1]),
          member(DR, [-1, 0, 1]),
          \+ (DC =:= 0, DR =:= 0),
          ToCol is Col + DC,
          ToRow is Row + DR,
          ToCol >= 1, ToCol =< 8,
          ToRow >= 1, ToRow =< 8,
          \+ friendly(Board, Color, ToCol, ToRow),
          \+ member(piece(_, king, ToCol, ToRow), Board),
          king_square_avoids_enemy_king(Board, Color, ToCol, ToRow),
          square_safe_for_king(Board, Color, ToCol, ToRow) ),
        Moves
    ).


%% ─── Dispatch ───────────────────────────────────────────────────────────────

%% piece_moves/6 – dispatches to the correct move generator based on piece type.
piece_moves(Board, Color, pawn, Col, Row, Moves) :-
    ( Color = white
    -> white_pawn_moves(Board, Col, Row, Moves)
    ;  black_pawn_moves(Board, Col, Row, Moves) ).

piece_moves(Board, Color, king, Col, Row, Moves) :-
    king_moves(Board, Color, Col, Row, Moves).


%% ─── All legal moves for a colour ───────────────────────────────────────────

%% legal_moves/3 – produces the flat list of all legal moves for Color on Board.
%% Each move is an internal move(FromCol, FromRow, ToCol, ToRow) term.
legal_moves(Board, Color, AllMoves) :-
    findall(
        MovesForPiece,
        ( member(piece(Color, Type, Col, Row), Board),
          piece_moves(Board, Color, Type, Col, Row, MovesForPiece) ),
        MoveLists
    ),
    flatten(MoveLists, AllMoves).

%% legal_moves_py/3 – Python-facing variant that returns each move as a
%% four-element integer list [FC, FR, TC, TR] rather than a compound term,
%% because Janus maps Prolog lists to Python lists natively but does not
%% automatically convert arbitrary compound terms to Python objects.
legal_moves_py(Board, Color, PyMoves) :-
    legal_moves(Board, Color, Moves),
    findall(
        [FC, FR, TC, TR],
        member(move(FC, FR, TC, TR), Moves),
        PyMoves
    ).


%% ─── Win conditions ─────────────────────────────────────────────────────────

%% win_condition/2 – true when Color has won the game on Board.
%% White wins by promoting a pawn to row 8, or by eliminating all black pawns.
%% Black wins by promoting a pawn to row 1, or by eliminating all white pawns.

win_condition(Board, white) :-
    %% A white pawn has reached the promotion rank.
    member(piece(white, pawn, _, 8), Board).

win_condition(Board, black) :-
    %% A black pawn has reached the promotion rank.
    member(piece(black, pawn, _, 1), Board).

win_condition(Board, white) :-
    %% Black has no pawns remaining.
    \+ member(piece(black, pawn, _, _), Board).

win_condition(Board, black) :-
    %% White has no pawns remaining.
    \+ member(piece(white, pawn, _, _), Board).
