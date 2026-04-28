import subprocess
import os

# === CONFIGURATION ===

DATASETS = [
    "com-youtube",
    "cit-Patents",
    "soc-LiveJournal1",
    "com-orkut",
    "eu-2015-host",
    "arabic-2005",
    "uk-2005",
    "twitter-2010",
    "com-friendster",
    "sk-2005"
]

ALPHA_LIST = [1.0] #[3.0, 4.0, 5.0] # [1.5, 2.0, 2.5]  # Power-law alphas
GEN_SCRIPT = "generate_weight_per_graph.py"


def get_powerlaw_output(graph_dir, dataset, alpha):
    return os.path.join(graph_dir, f"{dataset}_weight_a{alpha}.bin")


def get_degree_output(graph_dir, dataset):
    return os.path.join(graph_dir, f"{dataset}_weight_degree.bin")


# === MAIN LOOP ===

for dataset in DATASETS:
    print(f"\n[RUN] Processing dataset: {dataset}")
    graph_dir = f"/data/{dataset}"
    xadj_file = os.path.join(graph_dir, f"{dataset}_xadj.bin")
    edge_file = os.path.join(graph_dir, f"{dataset}_edge.bin")

    # -- Power-law loop
    for alpha in ALPHA_LIST:
        output_file = get_powerlaw_output(graph_dir, dataset, alpha)

        cmd = ["python3", GEN_SCRIPT,
               "--xadj", xadj_file,
               "--edge", edge_file,
               "--output", output_file,
               "--dist", "powerlaw",
               "--alpha", str(alpha)]

        print("[GEN] Generating powerlaw weights:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[OK] Saved: {output_file}")

    # -- Degree-based weights
    """
    output_file = get_degree_output(graph_dir, dataset)
    cmd = ["python3", GEN_SCRIPT,
           "--xadj", xadj_file,
           "--edge", edge_file,
           "--output", output_file,
           "--dist", "degree"]

    print("[GEN] Generating degree weights:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] Saved: {output_file}")
    """

