import numpy as np
import h5py
import os
import re


def h5_to_txt(h5_filename, txt_filename, num_problems=None):
    """
    Converts mesh-based CVRP data from H5 format to TXT format.

    Args:
        h5_filename: Path to the source H5 file
        txt_filename: Path to save the TXT file
        num_problems: Number of problems to convert (None for all)
    """
    with h5py.File(h5_filename, "r") as hf:
        # Get the number of problems in the file
        total_problems = hf["problems"].shape[0]
        if num_problems is None:
            num_problems = total_problems
        else:
            num_problems = min(num_problems, total_problems)

        one_row_data_all = []

        for i in range(num_problems):
            # Extract data for this problem
            problem = hf["problems"][i].tolist()
            demand = hf["demands"][i].tolist()
            capacity = int(hf["capacities"][i])
            distance = float(hf["distances"][i])
            node_flag = hf["node_flags"][i].tolist()

            demand = [int(d) for d in demand]

            # Get city coordinates (first entry is depot)
            city_indices = hf["city_indices"][:]
            problem_indices = city_indices[problem].tolist()

            problem_coords = [
                [problem[i] / 101, problem_indices[i] / 10001]
                for i in range(len(problem))
            ]

            # Separate depot and customer coordinates
            depot = problem_coords[0][:2]  # First 2 dimensions only
            customer_coords = [
                coord[:2] for coord in problem_coords[1:]
            ]  # Flatten to 1D list
            customer_coords_flat = [
                item for sublist in customer_coords for item in sublist
            ]

            # Create one row data format
            one_row_data = (
                ["depot"]
                + depot
                + ["customer"]
                + customer_coords_flat
                + ["capacity"]
                + [capacity]
                + ["demand"]
                + demand[1:]
                + ["cost"]
                + [distance]
                + ["node_flag"]
                + node_flag
            )

            one_row_data_all.append(one_row_data)

        # Save to TXT file
        np.savetxt(txt_filename, one_row_data_all, delimiter=",", fmt="%s")

    print(
        f"Successfully converted {num_problems} problems from {h5_filename} to {txt_filename}"
    )


def txt_to_h5(txt_filename, h5_filename, num_problems=100):
    """
    Converts CVRP data from TXT format to H5 format, storing actual 2D coordinates without padding or variable-length arrays.

    Assumption: All problems have the same number of customers.

    Args:
        txt_filename: Path to the source TXT file
        h5_filename: Path to save the H5 file
    """
    # Load the TXT data
    with open(txt_filename, "r") as f:
        lines = f.readlines()

    problems_coords = []  # depot + customer coords per problem
    demands = []
    capacities = []
    distances = []
    node_flags = []

    for line in lines[:num_problems]:
        parts = [p.strip().strip("\"'") for p in line.strip().split(",")]

        depot_idx = parts.index("depot")
        customer_idx = parts.index("customer")
        capacity_idx = parts.index("capacity")
        demand_idx = parts.index("demand")
        cost_idx = parts.index("cost")
        node_flag_idx = parts.index("node_flag")

        # Extract depot coordinates (2D)
        depot_coords = [float(parts[depot_idx + 1]), float(parts[depot_idx + 2])]

        # Extract customer coordinates (2D)
        customer_coords_flat = parts[customer_idx + 1 : capacity_idx]
        customer_coords = [
            [float(customer_coords_flat[i]), float(customer_coords_flat[i + 1])]
            for i in range(0, len(customer_coords_flat), 2)
        ]

        # Combine depot and customers into one coordinate list per problem
        problem_coords = [depot_coords] + customer_coords
        problems_coords.append(problem_coords)

        # Extract demands (first demand is depot=0)
        demand_values = [int(d) for d in parts[demand_idx + 1 : cost_idx]]
        demands.append(demand_values)

        # Extract capacity
        capacities.append(int(parts[capacity_idx + 1]))

        # Extract cost/distance
        distances.append(float(parts[cost_idx + 1]))

        # Extract node_flags
        node_flag_values = [int(nf) for nf in parts[node_flag_idx + 1 :]]
        node_flags.append(node_flag_values)

    num_problems = len(problems_coords)
    num_nodes_per_problem = len(problems_coords[0])  # assuming all same length

    # Convert lists into numpy arrays (fixed-size, no padding needed)
    coords_array = np.array(
        problems_coords, dtype=np.float32
    )  # shape: (num_problems, num_nodes, 2)
    demands_array = np.array(
        demands, dtype=np.int32
    )  # shape: (num_problems, num_nodes)
    capacities_array = np.array(capacities, dtype=np.int32)  # shape: (num_problems,)
    distances_array = np.array(distances, dtype=np.float32)  # shape: (num_problems,)
    node_flags_array = np.array(
        node_flags, dtype=np.int32
    )  # shape: (num_problems, num_nodes*2 - 2)

    # Save to H5 file with fixed-size datasets
    with h5py.File(h5_filename, "w") as hf:

        hf.create_dataset(
            "problems", data=coords_array
        )  # Actual coordinates stored here
        hf.create_dataset("demands", data=demands_array)
        hf.create_dataset("capacities", data=capacities_array)
        hf.create_dataset("distances", data=distances_array)
        hf.create_dataset("node_flags", data=node_flags_array)

    print(
        f"Successfully converted {num_problems} problems from {txt_filename} to {h5_filename}"
    )


# Example usage:
if __name__ == "__main__":
    # Convert from H5 to TXT
    # h5_to_txt("main_model/disk/problems/mesh_cvrp_data_train.h5", "vrp_train_data.txt")

    # Convert from TXT to H5 v
    txt_to_h5(
        "main_model/disk/problems/vrp100_hgs_train_100w.txt",
        "main_model/disk/problems/vrp100_hgs_train_100w.h5",
        num_problems=100,
    )
