import json
from itertools import product

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize

from vlm import VLM


# Load matplotlib settings
with open("config/matplotlib.json", "r") as file: plts = json.load(file)
plts["rcParams"]["text.latex.preamble"] = "".join(plts["latex_preamble"])
plt.rcParams.update(**plts["rcParams"])

# Load data
items_df = pd.read_csv("data/data.tsv", delimiter="\t", encoding="utf-8")\
             .sort_values("height", ascending=False)\
             .groupby(["width", "depth", "height"], as_index=False)\
             .aggregate({"instances": "sum"})

# Settings
GENERATE_COLUMNS_AND_LAYOUTS = False
PLOT_COLUMNS_AND_LAYOUTS     = False
PLOT_ITEMS_INSTANCES         = True
PLOT_ITEMS_SIZE_AND_CLUSTERS = False

LAYOUTS_DATA_PATH = "results/layouts_data.npy"
COLUMNS_DATA_PATH = "results/columns_data.npy"

MAX_TRAY_SIZE = np.array([1050, 1050, 325], dtype=np.int16)
MIN_TRAY_SIZE = np.array([100, 100, 75], dtype=np.int16)
SEPARATORS = np.array([20, 20, 0], dtype=np.int16)
GRAB_SPACE = np.array([10, 10, 0], dtype=np.int16)
DELTA = np.array([50, 50, 50], dtype=np.int16)
MAX_NUM_SECTORS = 3

# Items size and clusters
if PLOT_ITEMS_SIZE_AND_CLUSTERS: 
    vlm.items_scatter(
        color=np.argmax(vlm.is_item_in_cluster, axis=1), 
        figsize=(FIGSIZE_COL_WIDTH, 0.5*FIGSIZE_COL_WIDTH),
        cmap="twilight_shifted",
        save_path="plots/items_clusters.pdf"
    )

# Instances
if PLOT_ITEMS_INSTANCES:
    fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.3*plts["fig_width"]))
    ax.grid(which="both")
    ax.hist(items_df["instances"], log=True, density=True)
    ax.set_xlabel(r"$\theta_i$")
    ax.set_ylabel(r"(log) frequency [\%]")
    plt.savefig("plots/items_instances.pdf", bbox_inches="tight")
    plt.show(block=False)

# Columns and layouts generation 
if GENERATE_COLUMNS_AND_LAYOUTS:
    widths, depths = [np.arange(MIN_TRAY_SIZE[i], MAX_TRAY_SIZE[i] + DELTA[i], DELTA[i]) for i in range(2)]
    columns_data = np.zeros(depths.size, dtype=[("time", "<f4"), ("n", "<f4")])
    layouts_data = np.zeros((widths.size, depths.size), dtype=[("time", "<f4"), ("n", "<f4")])

    for i,depth in enumerate(depths):
        vlm = VLM(tray=np.array([1, depth, 1], dtype=np.int16), delta=DELTA, min_sector=MIN_TRAY_SIZE, separators=SEPARATORS,
                  grab_space=GRAB_SPACE, items_df=items_df, max_num_sectors=MAX_NUM_SECTORS, max_height_distance=10, process=False)
        columns_results = vlm.get_columns()

        columns_data["time"][i] = columns_results.statistics["time"].total_seconds()
        columns_data["n"][i] = columns_results.statistics["nSolutions"]
    np.save("results/columns_data.npy", columns_data, allow_pickle=False)

    for width, depth in product(widths, depths):
        print(f"Tray width: {width}x{depth}")
        row, col = (width - MIN_TRAY_SIZE[0]) // DELTA[0], (depth - MIN_TRAY_SIZE[1]) // DELTA[1]
        vlm = VLM(tray=np.array([width, depth, 1], dtype=np.int16), delta=DELTA, min_sector=MIN_TRAY_SIZE, separators=SEPARATORS,
                  grab_space=GRAB_SPACE, items_df=items_df, max_num_sectors=MAX_NUM_SECTORS, max_height_distance=10, process=False)
        layouts_results = vlm.get_layouts()

        layouts_data["time"][row, col] = layouts_results.statistics["time"].total_seconds()
        layouts_data["n"][row, col] = layouts_results.statistics["nSolutions"]
    np.save("results/layouts_data.npy", layouts_data, allow_pickle=False)

if PLOT_COLUMNS_AND_LAYOUTS:
    columns_data = np.load(COLUMNS_DATA_PATH)
    layouts_data = np.load(LAYOUTS_DATA_PATH)

    # Plot columns data
    fig, axs = plt.subplots(1,2, figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
    label=["Computation time [s]", r"$|\mathcal{Z}|$"]
    ticks = np.arange(0, columns_data.shape[0], 3)
    
    for i,field in enumerate(columns_data.dtype.fields.keys()):
        axs[i].bar(np.arange(columns_data[field].size), columns_data[field])
        axs[i].set_xticks(ticks, labels=[str(MIN_TRAY_SIZE[1] // DELTA[0] + i) for i in ticks])
        axs[i].set_xlabel(r"$\overline{q}_2$")
        axs[i].set_ylabel(label[i])
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,-3))
        axs[i].grid()
        
    plt.subplots_adjust(wspace=0.4)
    plt.savefig("plots/columns.pdf", bbox_inches="tight")
    plt.show(block=False)

    # Plot layouts data
    fig, axs = plt.subplots(1,2, figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
    label=["Computation time [s]", r"$|\mathcal{L}| / Q_3$"]
    ticks = np.arange(0, layouts_data.shape[0], 3)
    axs[0].set_ylabel(r"$\overline{q}_2$")

    for i,field in enumerate(layouts_data.dtype.fields.keys()):
        im = axs[i].imshow(layouts_data[field], cmap="Blues", interpolation="nearest")
        axs[i].set_xlabel(r"$\overline{q}_2$")
        axs[i].spines[['right', 'top']].set_visible(True)

        axs[i].set_xticks(ticks, labels=[str(MIN_TRAY_SIZE[0] // DELTA[0] + i) for i in ticks])
        axs[i].set_yticks(ticks, labels=[str(MIN_TRAY_SIZE[1] // DELTA[0] + i) for i in ticks])
        cbar = plt.colorbar(im, ax=axs[i], location='top', anchor=(0.5, 0.8), shrink=0.85)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label(label[i])
        cbar.ax.set_xlim(layouts_data[field].min(), layouts_data[field].max())

    plt.savefig("plots/layouts.pdf", bbox_inches="tight")
    plt.show(block=False)




# vlm = VLM(
#     tray=np.array([900, 600, 325], dtype=np.int16),
#     delta=np.array([50, 50, 50], dtype=np.int16),
#     min_sector=np.array([100, 100, 75], dtype=np.int16),
#     separators=np.array([20, 20, 0], dtype=np.int16),
#     grab_space=np.array([10, 10, 0], dtype=np.int16),
#     items_df=items_df,
#     max_num_sectors=2
# )















# # With NextowkX
# item_cluster_graph = nx.bipartite.from_biadjacency_matrix(sp.sparse.coo_matrix(is_item_in_cluster))
# nx.draw_networkx(
#     item_cluster_graph, 
#     pos=nx.planar_layout(item_cluster_graph), 
#     with_labels=False, 
#     node_size=10,
#     node_color=["#1f78b4"] * is_item_in_cluster.shape[0] + ["#ff7f0e"] * is_item_in_cluster.shape[1],
#     width=0.1
# )
# plt.show(block=False)

# # Export item-item_cluster pairs
# num_items_in_cluster = is_item_in_cluster.sum(0)
# item_cluster_df = pd.DataFrame({"num_items_in_cluster": num_items_in_cluster})
# tm = px.treemap(item_cluster_df, values="num_items_in_cluster")
# tm.show()


# item_cluster_pairs = np.vstack([np.arange(is_item_in_cluster.shape[0]), np.argmax(is_item_in_cluster, axis=1)]).T
# np.savetxt("external/item_cluster_pairs.csv", item_cluster_pairs, delimiter=",", fmt="'%d'")

# # Plotting
# fig = plt.figure(figsize=(FIGSIZE_COL_WIDTH, FIGSIZE_COL_WIDTH))
# ax = fig.add_subplot(projection="3d")
# ax.set_aspect(aspect="auto", anchor="SW")
# # ax.set_box_aspect(aspect=None, zoom=0.8)

# background_color = (1.0, 1.0, 1.0, 0.0)

# ax.xaxis.set_pane_color(background_color)
# ax.yaxis.set_pane_color(background_color)
# ax.zaxis.set_pane_color(background_color)

# ax.set_xlabel("item width [mm]", labelpad=-1)
# ax.set_ylabel("item depth [mm]", labelpad=-1)
# ax.set_zlabel("item height [mm]", labelpad=-1)

# ax.tick_params(axis="y", pad=0)
# ax.tick_params(axis="z", pad=0)
# ax.tick_params(axis="x", pad=0)

# # ax.imshow(c_item_fits_c_sector, cmap="binary")
# ax.scatter(*[items_size[:,i] for i in range(3)], s=3, c=np.argmax(is_item_in_cluster, axis=1), cmap="twilight_shifted")

# # plt.tight_layout()
# plt.savefig("plots/clustered_item.pdf")
# plt.show(block=False)

# for i,attr in enumerate(("width", "depth", "height")):
#     fig, ax = plt.subplots(figsize=(FIGSIZE_COL_WIDTH, 0.33*FIGSIZE_COL_WIDTH))
#     n, bins, patches = ax.hist(items_df[attr].to_numpy(), 50, density=True)
#     ax.set_xlabel(f"{attr} ($i_{i+1}$) [mm]")
#     ax.set_ylabel("log-freq. (\%)")
#     plt.yscale("log")
#     ax.grid(alpha=0.2)
#     # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     plt.tight_layout()
#     plt.savefig(f"plots/items_{attr}_distr.pdf", bbox_inches="tight")

# plt.show(block=False)



