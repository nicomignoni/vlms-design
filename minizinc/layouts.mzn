include "count_leq.mzn";

% Data
1..: MAX_NUM_SECTORS;
array[1..3] of 1..: TRAY;
array[_, 1..2] of 1..: TILES;
array[_, 1..3] of 1..: COLUMNS; % num_nonzeros x 3 [row: (column, tile, instances)]
array[_] of 1..: COLUMNS_WIDTH; % num_columns x 1

% Variables
array[index_set_1of2(TILES)] of var 0..: layout;
array[index_set(COLUMNS_WIDTH)] of var 0..: col_instances ::no_output;

% The overall tiles area must equal the tray area
constraint sum(t in index_set_1of2(TILES)) (layout[t] * TILES[t,1] * TILES[t,2]) = TRAY[1] * TRAY[2];

% Columns must fill the tray along the width
constraint sum(c in index_set(COLUMNS_WIDTH)) (col_instances[c] * COLUMNS_WIDTH[c]) = TRAY[1];

% Preserve the number of used tiles
constraint forall(t in index_set_1of2(TILES)) (
    layout[t] = sum(k in index_set_1of2(COLUMNS) where COLUMNS[k,2] = t) (col_instances[COLUMNS[k,1]] * COLUMNS[k,3])
);

% Limit the maximum number of used tiles
constraint count_leq(layout, 0, length(layout) - MAX_NUM_SECTORS);

solve satisfy;



