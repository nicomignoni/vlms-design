include "count_leq.mzn";

% Data
1..: NUM_SECTORS;
1..: MAX_NUM_LAYOUTS;
1..: MAX_NUM_TRAYS;
1..: VLM_HEIGHT;
1..: SAFETY_GAP;
array[_] of 1..: ITEMS; 
array[_] of 1..: LAYOUTS_HEIGHT;
array[_, 1..3] of 1..: LAYOUTS; % num_nonzeros x 3 [row: (layout, sector, instances)]
array[_, 1..2] of 1..: FIT; % num_nonzeros x 2 [row: (item, sector)]

% Variables
array[index_set_1of2(FIT)] of var 0..100: assignment;
array[index_set(LAYOUTS_HEIGHT)] of var 0..100: trays;

% Max number of layouts and trays
constraint sum(l in index_set(trays)) (trays[l]) <= MAX_NUM_TRAYS;
constraint count_leq(trays, 0, length(trays) - MAX_NUM_LAYOUTS);

% Machine height
% constraint sum(l in index_set(trays)) (trays[l] * (LAYOUTS_HEIGHT[l] + SAFETY_GAP)) <= VLM_HEIGHT;

% Used sectors and allocated items
constraint forall(s in 1..NUM_SECTORS) (
    sum(j in index_set_1of2(FIT) where FIT[j,2] = s) (assignment[j]) <= 
    sum(j in index_set_1of2(LAYOUTS) where LAYOUTS[j,2] = s) (LAYOUTS[j,3] * trays[LAYOUTS[j,1]])
);
constraint forall(i in index_set(ITEMS)) (
    sum(j in index_set_1of2(FIT) where FIT[j,1] = i) (assignment[j]) = ITEMS[i]
);

solve minimize sum(l in index_set(trays)) (trays[l] * LAYOUTS_HEIGHT[l]);
