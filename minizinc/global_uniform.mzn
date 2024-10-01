% Data
1..: MAX_NUM_SECTORS;
1..: MAX_NUM_LAYOUTS;
1..: MAX_NUM_TRAYS;
1..: VLM_HEIGHT;
1..: SAFETY_GAP;
array[1..3] of 1..: TRAY;
array[_] of 1..: ITEMS;
array[_, 1..3] of 1..: SECTORS;
array[_, 1..2] of 1..: FIT; % row: nonzeros, col: [item, sector]

% Expressions
set of 1..: HEIGHTS = array2set(col(SECTORS,3));

% Variables
array[1..MAX_NUM_LAYOUTS] of var 0..: layout_trays;
array[1..MAX_NUM_LAYOUTS] of var HEIGHTS: layout_height;
array[1..MAX_NUM_LAYOUTS, index_set_1of2(FIT)] of var 0..: layout_items;
array[1..MAX_NUM_LAYOUTS, index_set_1of2(SECTORS)] of var 0..: layout_sectors;

% Allocation constraints
constraint forall(i in index_set(ITEMS)) (
    ITEMS[i] = sum(l in 1..MAX_NUM_LAYOUTS, j in index_set_1of2(FIT) where FIT[j,1] = i) (layout_items[l,j])
);
constraint forall(s in index_set_1of2(SECTORS)) (
    sum(l in 1..MAX_NUM_LAYOUTS) (layout_sectors[l,s] * layout_trays[l]) = 
    sum(l in 1..MAX_NUM_LAYOUTS, j in index_set_1of2(FIT) where FIT[j,2] = j) (layout_items[l,j])
);

% Sectors and trays limits
constraint sum(l in 1..MAX_NUM_LAYOUTS) (layout_trays[l]) <= MAX_NUM_TRAYS;
constraint forall(l in 1..MAX_NUM_LAYOUTS) (sum(s in index_set_1of2(SECTORS)) (layout_sectors[l,s] >= 1) <= MAX_NUM_SECTORS);

% Geometrical conditions
constraint sum(l in 1..MAX_NUM_LAYOUTS) (layout_height[l] + SAFETY_GAP) <= VLM_HEIGHT;
constraint forall(l in 1..MAX_NUM_LAYOUTS) (sum(s in index_set_1of2(SECTORS)) (layout_sectors[l,s] * SECTORS[s,2] mod TRAY[2]) = 0);
constraint forall(l in 1..MAX_NUM_LAYOUTS) (sum(s in index_set_1of2(SECTORS)) (layout_sectors[l,s] * SECTORS[s,2] div TRAY[2]) = TRAY[1]);
constraint forall(l in 1..MAX_NUM_LAYOUTS, s in index_set_1of2(SECTORS) where layout_sectors[l,s] >= 1) (layout_height[l] = SECTORS[s,3]);

solve satisfy;