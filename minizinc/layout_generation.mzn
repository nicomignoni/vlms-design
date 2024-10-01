% Data
int: TRAY_WIDTH;
int: TRAY_DEPTH;

int: TILES_MAX_WIDTH; 
int: TILES_MAX_DEPTH; 

int: TILES_MIN_WIDTH;
int: TILES_MIN_DEPTH;

int: MAX_NUM_SECTORS;

array[_] of int: TILES_AREA;
array[_] of int: MAX_NUM_PLACEABLE_TILES;
array[_, 1..2] of int: TILES_SIZE;

% Expressions
int: NUM_TILES = (TILES_MAX_WIDTH - TILES_MIN_WIDTH + 1) * (TILES_MAX_DEPTH - TILES_MIN_DEPTH + 1);
int: TRAY_AREA = TRAY_WIDTH * TRAY_DEPTH;
int: MAX_NUM_COLUMNS = TRAY_WIDTH div TILES_MIN_WIDTH;

% Variables
array[1..NUM_TILES] of var 0..: layout;

% Auxiliary variables
array[1..MAX_NUM_COLUMNS, 1..NUM_TILES] of var 0..: columns ::no_output;
array[1..MAX_NUM_COLUMNS] of var {0} union TILES_MIN_WIDTH..TRAY_WIDTH: widths ::no_output;

% Bound number of tiles
constraint forall(t in 1..NUM_TILES) (layout[t] in 0..MAX_NUM_PLACEABLE_TILES[t]);

% Area constraint
constraint sum(t in 1..NUM_TILES) (layout[t] * TILES_AREA[t]) = TRAY_AREA;

% Impose that the length of stacked columns is a multiple of the tray depth
constraint redundant_constraint(sum(t in 1..NUM_TILES) (layout[t] * TILES_SIZE[t,2]) mod TRAY_DEPTH = 0);

% Bound number of sectors
constraint NUM_TILES - count(layout,0) <= MAX_NUM_SECTORS;

% Preserve the number of tiles
constraint forall(t in 1..NUM_TILES) (sum(c in 1..MAX_NUM_COLUMNS) (columns[c,t]) = layout[t]);

% Fill the column along the depth direction and fix tiles width to be all equal
constraint forall(c in 1..MAX_NUM_COLUMNS) (
    widths[c] >= 1 <-> (sum(t in 1..NUM_TILES) (columns[c,t] * TILES_SIZE[t,2]) = TRAY_DEPTH /\
                        forall(t in 1..NUM_TILES where columns[c,t] >= 1) (TILES_SIZE[t,1] = widths[c]))
);

% If the column has no width, no tile is deployed 
constraint forall(c in 1..MAX_NUM_COLUMNS) (
    widths[c] = 0 <-> sum(t in 1..NUM_TILES) (columns[c,t]) = 0
);

% Fill the tray along the width
constraint sum(c in 1..MAX_NUM_COLUMNS) (widths[c]) = TRAY_WIDTH;

solve satisfy;