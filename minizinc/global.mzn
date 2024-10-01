% Data
1..: MAX_NUM_SECTORS;
1..: MAX_NUM_LAYOUTS;
1..: MAX_NUM_TRAYS;
1..: VLM_HEIGHT;
1..: SAFETY_GAP;

array[1..3] of 1..: TRAY_SIZE;
array[_] of 1..: C_ITEMS_INSTANCES;
array[_, 1..3] of 1..: SECTORS_SIZE;
array[_, _] of bool: C_ITEM_FITS_C_SECTOR;
array[_, _] of bool: IS_SECTOR_IN_C_SECTOR;

set of 1..: HEIGHTS = array2set(col(SECTORS_SIZE,3));

% Variables
array[1..MAX_NUM_LAYOUTS] of var 0..: trays;
array[1..MAX_NUM_LAYOUTS] of var HEIGHTS: layout_height;
array[1..MAX_NUM_LAYOUTS, index_set_1of2(SECTORS_SIZE)] of var 0..: layout_sectors;
array[index_set_1of2(C_ITEM_FITS_C_SECTOR), index_set_2of2(C_ITEM_FITS_C_SECTOR)] of var 0..: allocation;

% Limit the maximum number of usable trays
% constraint sum(trays) <= MAX_NUM_TRAYS;

% % Limit the maximum  number of usable sectors, per layout
% constraint forall(l in 1..MAX_NUM_LAYOUTS) (
%     sum(s in index_set_1of2(SECTORS_SIZE)) (layout_sectors[l,s] >= 1) <= MAX_NUM_SECTORS
% );

% % Limit the height of the trays stack
% constraint sum(l in 1..MAX_NUM_LAYOUTS) (trays[l] * (layout_height[l] + SAFETY_GAP)) <= VLM_HEIGHT;

% % Active layouts have columns filling the tray width, and all used sectors have the same height
% constraint forall(l in 1..MAX_NUM_LAYOUTS) (
%     trays[l] >= 1 <-> sum(s in index_set_1of2(SECTORS_SIZE)) (SECTORS_SIZE[s,1] * TRAY_SIZE[2] div layout_sectors[l,s]) = TRAY_SIZE[1] /\
%                      forall(s in index_set_1of2(SECTORS_SIZE) where layout_sectors[l,s] >= 1) (layout_height[l] = SECTORS_SIZE[s,3])
% );

% % Non active layouts don't have any sector
% constraint forall(l in 1..MAX_NUM_LAYOUTS) (
%     trays[l] = 0 <-> sum(s in index_set_1of2(SECTORS_SIZE)) (layout_sectors[l,s]) = 0
% );

% % All instances of each item cluster must be allocated
% constraint forall(ic in index_set_1of2(C_ITEM_FITS_C_SECTOR)) (
%     sum(sc in index_set_2of2(C_ITEM_FITS_C_SECTOR)) (allocation[ic,sc]) = C_ITEMS_INSTANCES[ic]
% );

% % The number of sectors in each sector cluster must equal the total number of available sectors belonging to said cluster
% constraint forall(sc in index_set_2of2(C_ITEM_FITS_C_SECTOR)) (
%     sum(ic in index_set_1of2(C_ITEM_FITS_C_SECTOR)) (allocation[ic,sc]) =
%     sum(l in 1..MAX_NUM_LAYOUTS, s in index_set_1of2(SECTORS_SIZE) where IS_SECTOR_IN_C_SECTOR[s,sc] = true) (trays[l] * layout_sectors[l,s])
% );

% % Area constraint as necessary condition
% constraint forall(l in 1..MAX_NUM_LAYOUTS) (
%     sum(s in index_set_1of2(SECTORS_SIZE)) (SECTORS_SIZE[s,1] * SECTORS_SIZE[s,2] * layout_sectors[l,s]) = TRAY_SIZE[1] * TRAY_SIZE[2]
% );

solve satisfy;
