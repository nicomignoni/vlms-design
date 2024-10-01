include "count_leq.mzn";

% Data
0..: MAX_NUM_SECTORS; 
array[_] of 1..: DEPTHS;
array[1..3] of 1..: TRAY; 

% Variables
array[index_set(DEPTHS)] of var 0..: column;

% Used tiles must fill the column depth (2)
constraint sum(t in index_set(DEPTHS)) (DEPTHS[t] * column[t]) == TRAY[2];

% Limit the number of used tiles
constraint count_leq(column, 0, length(column) - MAX_NUM_SECTORS);

solve satisfy; 