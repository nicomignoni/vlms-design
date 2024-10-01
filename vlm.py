from itertools import product, permutations

import pandas as pd
import numpy as np
import scipy as sp
import cvxpy as cp

from minizinc import Model, Solver, Instance

import matplotlib.pyplot as plt 


class VLM:
    def __init__(self, tray, delta, min_sector, max_num_sectors, max_height_distance,
                 separators, grab_space, items_df, process=True):
        self.tray = tray
        self.delta = delta
        self.min_sector = min_sector
        self.max_num_sectors = max_num_sectors
        self.separators = separators
        self.grab_space = grab_space
        self.max_height_distance = max_height_distance
        self.items_df = items_df

        self.sizes = [
            np.arange(self.min_sector[i], self.tray[i] + self.delta[i], self.delta[i], dtype=np.int16) 
            for i in range(3)
        ]
        self.tiles = np.fromiter(product(*self.sizes[:2]), dtype="(2,)int")
        self.sectors = np.fromiter(product(*self.sizes), dtype="(3,)int")

        if process:
            # Items/sectors fitting
            perms = np.fromiter(permutations(range(3)), count=6, dtype="(3,)int")
            self.item_fits_sector = np.logical_and(
                # Basic condition
                (
                    self.items_df[["width", "depth", "height"]].values[:,None,perms] <= \
                    self.sectors[None,:,None,:] - \
                    self.separators[None,None,None,:] - \
                    self.grab_space[None,None,None,:]
                ).all(-1).any(-1),
                # Height distance betweem items and sectors is less them max_height_distance
                self.sectors[:,2] - self.items_df["height"].values[:,None] <= self.max_height_distance
            )

            # Group together item that can be placed in the same sectors
            _, unique_items, items_cluster = np.unique(self.item_fits_sector, axis=0, return_index=True, return_inverse=True)
            self.is_item_in_cluster = np.equal(items_cluster[:,None], unique_items)
            print(f"{items_cluster.size} item types -> {unique_items.size} item clusters")

            # Group together sector that can be hold the same items
            _, unique_sectors, sectors_cluster = np.unique(self.item_fits_sector, axis=1, return_index=True, return_inverse=True)
            self.is_sector_in_cluster = np.equal(sectors_cluster[:,None], unique_sectors)
            print(f"{sectors_cluster.size} sector types -> {unique_sectors.size} sector clusters")

            # Items cluster/sectors cluster fitting and items clusters instances
            self.c_item_fits_c_sector = self.item_fits_sector[unique_items][:,unique_sectors]
            self.c_items_instances = self.items_df["instances"].values @ self.is_item_in_cluster


    def __repr__(self):
        return f"VLM, tray: {'x'.join(map(str, self.tray))} mm, " +\
               f"delta: {'x'.join(map(str, self.delta))}"

    # Plotting utilities
    def items_scatter(self, color, save_path, **params):
        if "norm" not in params: params["norm"] = None
        fig = plt.figure(figsize=params["figsize"])
        ax = fig.add_subplot()

        ax.set_xlabel(r"$\sigma^{\mathcal{C}}_{i,1}$ [mm]")
        ax.set_ylabel(r"$\sigma^{\mathcal{C}}_{i,2}$ [mm]")
        ax.grid(zorder=0)

        ax.scatter(
            x=self.items_df["width"], y=self.items_df["depth"],
            s=50 * self.items_df["height"] / self.items_df["height"].max(), 
            c=color, ec="w", lw=1e-5, cmap=params["cmap"], norm=params["norm"]
        )
        
        if save_path: plt.savefig(save_path, bbox_inches="tight")
        plt.show(block=False)

    
    def heuristic_get_sectors_assignment(self):
        sectors_volume = np.prod(self.sectors, axis=1)
        sectors_height_weight = self.sectors[:,2] / self.sectors[:,2].max()
        sectors_score = 1 / np.log(sectors_height_weight * sectors_volume)
        selected_sectors = np.argmax(self.is_sector_in_cluster.T * sectors_score, axis=1)

        assignment = cp.Variable(self.c_item_fits_c_sector.shape, integer=True)
        constraints = [cp.sum(assignment, axis=1) == self.c_items_instances]
        objective = cp.sum(assignment, 0) @ sectors_score[sectors_score]

        prob = cp.Problem(cp.Maximize(objective), constraints)
        return prob.solve(solver=cp.SCIPY, scipy_options={"disp": True})


    def get_columns(self, mzn_file="minizinc/columns.mzn"):
        print("Generating columns...")

        # Columns generation model
        gecode = Solver.lookup("gecode")
        columns_model = Model(mzn_file)
        columns_instance = Instance(gecode, columns_model)

        columns_instance["TRAY"] = self.tray
        columns_instance["DEPTHS"] = self.sizes[1]
        columns_instance["MAX_NUM_SECTORS"] = self.max_num_sectors

        columns_results = columns_instance.solve(all_solutions=True, processes=4)
        print(
            f"Flat columns generation done ({columns_results.statistics['time'].total_seconds()} sec.),",
            f"{columns_results.statistics['nSolutions']} generated depth partitions)."
        )
        return columns_results
        

    def get_layouts(self, mzn_file="minizinc/layouts.mzn"):
        print("Generating layouts...")

        # Get columns
        columns_results = self.get_columns()
        columns_tiles = np.kron(
            np.eye(self.sizes[0].size, dtype=np.int16),
            np.array([sol.column for sol in columns_results.solution], dtype=np.int16),
        )
        columns_width = np.kron(self.sizes[0], np.ones(columns_results.statistics["nSolutions"], dtype=np.int16))
        print(f"(Depth) columns: {columns_results.statistics['nSolutions']} ->",
              f"(Flat) columns {columns_tiles.shape[0]}")

        nonzero_columns_rows, nonzero_columns_cols = columns_tiles.nonzero()

        # Layout generation model
        gecode = Solver.lookup("gecode")
        layouts_model = Model(mzn_file)
        layouts_instance = Instance(gecode, layouts_model)

        layouts_instance["MAX_NUM_SECTORS"] = self.max_num_sectors
        layouts_instance["COLUMNS_WIDTH"] = columns_width
        layouts_instance["TILES"] = self.tiles
        layouts_instance["TRAY"] = self.tray
        layouts_instance["COLUMNS"] = np.vstack([
            nonzero_columns_rows + 1, 
            nonzero_columns_cols + 1, 
            columns_tiles[nonzero_columns_rows, nonzero_columns_cols]
        ]).T

        layout_results = layouts_instance.solve(all_solutions=True, processes=4)
        print(f"Flat layouts generation done ({layout_results.statistics['time'].total_seconds()} sec., ",
              f"{layout_results.statistics['nSolutions']} generated flat layouts).")

        return layout_results

    def get_assignments(
            self, max_num_layouts, max_num_trays, safety_gap, max_num_height, 
            vlm_height, mzn_file=r"minizinc/assignment.mzn"
        ):
        print("Generating assignment...")

        # Get layouts
        layouts_results = self.get_layouts()
        layouts_sectors = sp.sparse.kron(
            sp.sparse.csr_array([sol.layout for sol in layouts_results.solution], dtype=np.int16),
            sp.sparse.eye(self.sizes[2].size, dtype=np.int16)
        )
        layouts_heights = np.kron(np.ones(layouts_results.statistics["nSolutions"], dtype=np.int16), self.sizes[2]) 
        print(f"(Flat) layouts: {layouts_results.statistics['nSolutions']} ->",
              f"(Sector) layouts: {layouts_sectors.shape[0]}")
        
        # Reduce, layout matrices, and items
        layouts_c_sectors = layouts_sectors @ sp.sparse.csr_matrix(self.is_sector_in_cluster)

        # Group together layouts having the same sectors clusters
        c_layouts_c_sectors, unique_layout_clusters, layout_clusters = np.unique(
            layouts_c_sectors.todense(), axis=0, return_index=True, return_inverse=True
        )
        is_layout_in_cluster = np.equal(layout_clusters[:,None], unique_layout_clusters)
        print(f"{layouts_c_sectors.shape[0]} layouts -> {unique_layout_clusters.size} layouts clusters")

        # # Sparsify matrices
        # c_item_fits_c_sector_indices = c_item_fits_c_sector.nonzero()
        # print(f"c_item_fits_c_sector: {c_item_fits_c_sector_indices[0].size}/{c_item_fits_c_sector.size}",
        #       f"({(c_item_fits_c_sector_indices[0].size / c_item_fits_c_sector.size):.2f})")

        # u_layouts_c_sectors_indices = u_layouts_c_sectors.nonzero()
        # print(f"layouts_sectors: {u_layouts_c_sectors_indices[0].size}/{u_layouts_c_sectors.size}",
        #       f"({(u_layouts_c_sectors_indices[0].size / u_layouts_c_sectors.size):.2f})")

        # The virtual height of a layout cluster is the minimum height of the layouts there contained
        c_layouts_height = np.minimum(is_layout_in_cluster * layouts_heights[:,None], axis=0)
        has_c_layout_height = np.sparse.csr(np.equal(layouts_heights[:,None], c_layouts_height))

        
        # Assignment model
        assignment = cp.Variable(self.c_item_fits_c_sector.shape, integer=True)
        trays_per_c_layout = cp.Variable(c_layouts_c_sectors.shape[0], integer=True)
        is_c_layout_used = cp.Variable(c_layouts_c_sectors.shape[0], boolean=True)
        is_c_height_used = cp.Variable(self.sizes[2].size, boolean=True)
        
        total_c_height = trays_per_c_layout @ c_layouts_height
        total_layouts = cp.sum(is_c_layout_used)
        total_trays_per_c_layout = cp.sum(trays_per_c_layout)
        total_c_height = cp.sum(is_c_height_used)
        total_c_layouts_with_height = has_c_layout_height.T @ is_c_layout_used
        max_num_trays_per_c_layout = vlm_height // c_layouts_height

        constraints = [
            # Variables bounds
            trays_per_c_layout >= 0,
            assignment >= 0,
            assignment <= self.c_items_instances[:,None] * self.c_item_fits_c_sector,

            # Logical conditions
            trays_per_c_layout >= is_c_layout_used,
            trays_per_c_layout <= cp.multiply(is_c_layout_used, max_num_trays_per_c_layout),
            total_c_layouts_with_height >= is_c_height_used,
            total_c_layouts_with_height <= cp.multiply(is_c_height_used, has_c_layout_height.T @ max_num_trays_per_c_layout),

            # Geometrical constraints
            total_trays_per_c_layout <= max_num_trays,
            total_layouts <= max_num_layouts,
            trays_per_c_layout @ (c_layouts_height + safety_gap) <= vlm_height,
            total_c_height <= max_num_height,

            # Assignment constraints
            cp.sum(assignment, 1) == self.c_items_instances,
            cp.sum(assignment, 0) == trays_per_c_layout @ c_layouts_c_sectors
        ]
        
        prob = cp.Problem(cp.Minimize(total_c_height), constraints)

        return prob.solve(solver=cp.SCIPY, scipy_options={"disp": True})
        

if __name__ == "__main__":
    items_df = pd.read_csv("data/data.tsv", delimiter="\t", encoding="utf-8")

    vlm = VLM(
        items_df=items_df,
        tray=np.array([900, 600, 325], dtype=np.int16),
        delta=np.array([50, 50, 50], dtype=np.int16),
        min_sector=np.array([100, 100, 75], dtype=np.int16),
        separators=np.array([20, 20, 0], dtype=np.int16),
        grab_space=np.array([10, 10, 0], dtype=np.int16),
        max_num_sectors=3,
        max_height_distance=10
    )

    print(vlm)

    a = vlm.get_assignments(
        max_num_layouts=40,
        max_num_trays=500,
        safety_gap=25,
        vlm_height=54300,
        max_num_height=4
    )