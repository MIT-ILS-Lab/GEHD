import os
import time
import multiprocessing
from functools import partial
import h5py
import logging
import numpy as np
from tqdm import tqdm
import trimesh
import multiprocessing as mp

from pygeodesic import geodesic
from pyvrp import Model
from pyvrp.stop import MaxRuntime

from main_model.src.utils.general_utils import parse_pyvrp_solution

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Send logs to the console (stdout)
        # logging.FileHandler("my_log_file.log"),  # Optionally, write logs to a file
    ],
)

_geoalg = None
_source_indices = None


def init_worker(vertices, faces, source_indices):
    """Initialize the worker process with the geodesic algorithm."""
    global _geoalg, _source_indices
    from pygeodesic import (
        geodesic,
    )  # Import inside the function to ensure it's imported in the worker process

    _geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    _source_indices = source_indices


def worker_function(i):
    """Worker function that uses the pre-initialized geodesic algorithm."""
    global _geoalg, _source_indices
    index_arr = np.array([_source_indices[i]])
    distances = np.array(_geoalg.geodesicDistances(index_arr, _source_indices))[0]
    return i, distances


def compute_geodesic_distances_parallel(
    vertices, faces, source_indices, num_processes=None
):
    """Compute geodesic distances using parallel processing."""
    # Set default number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Create empty matrix to store results
    geodesic_mat = np.zeros((len(source_indices), len(source_indices)))

    worker_iteration = [i for i in range(len(source_indices))]

    start_time = time.time()
    # Create a pool of workers with initialization
    with mp.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(vertices, faces, source_indices),
    ) as pool:
        # Use imap with tqdm to show progress
        results = list(
            tqdm(
                pool.imap(worker_function, worker_iteration),
                total=len(source_indices),
                desc="Computing geodesic distances",
            )
        )

    # Fill the matrix with results
    for i, distances in results:
        geodesic_mat[i, :] = distances

    end_time = time.time()
    elapsed_time = end_time - start_time
    # add time taken in minutes
    logging.info(f"Time taken to compute geodesic distances: {elapsed_time/60:.2f}m")

    return geodesic_mat


def transform_solution(solution):
    """
    Converts a one-row solution format to a two-row format with nodes and flags.

    Args:
        solution: Original solution as a list with depot (0) separating routes

    Returns:
        List containing nodes followed by flags (1 for start of route, 0 otherwise)
    """
    node = []
    flag = []
    for i in range(1, len(solution)):
        if solution[i] != 0:
            node.append(solution[i])
        if solution[i] != 0 and solution[i - 1] == 0:
            flag.append(1)
        if solution[i] != 0 and solution[i - 1] != 0:
            flag.append(0)
    node_flag = node + flag
    return node_flag


def get_mesh_city(mesh_path: str, num_customers: int, seed: int = 0):
    """
    Generates a city for the CVRP with a single fixed depot on a trimesh.

    Args:
        mesh_path: Path to the mesh file (.obj, .ply, etc.)
        num_customers: The number of customers to sample on the mesh
        seed: Random seed for reproducibility

    Returns:
        A dictionary containing mesh data and sampled points
        The city size (number of customers + depot)
    """
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Load the mesh
    logging.info(f"Loading mesh from {mesh_path}")
    mesh = trimesh.load_mesh(mesh_path)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Select depot (using vertex closest to the centroid)
    centroid = mesh.centroid
    depot_idx = np.argmin(np.linalg.norm(vertices - centroid, axis=1))
    logging.info(f"Selected depot at vertex index {depot_idx}")

    # Sample customer locations from mesh vertices
    available_indices = np.arange(len(vertices))
    available_indices = available_indices[
        available_indices != depot_idx
    ]  # Exclude depot

    if (num_customers == -1) or (num_customers > len(available_indices)):
        city_size = len(available_indices)
    else:
        city_size = num_customers

    customer_indices = np.random.choice(available_indices, city_size, replace=False)
    logging.info(f"Sampled {city_size} customer locations from mesh vertices")

    # Combine depot and customers
    city_indices = np.concatenate([[depot_idx], customer_indices])
    city = vertices[city_indices]

    # Compute geodesic distances between all points
    logging.info("Computing geodesic distance matrix...")
    geodesic_matrix = np.zeros((len(city_indices), len(city_indices)), dtype=np.float32)
    geodesic_matrix = compute_geodesic_distances_parallel(vertices, faces, city_indices)
    logging.info("Finished computing geodesic distance matrix")

    return {
        "mesh": mesh,
        "vertices": vertices,
        "faces": faces,
        "city": city,
        "city_indices": city_indices,
        "geodesic_matrix": geodesic_matrix,
    }, city_size


def get_problem(
    city_size: int,
    problem_size: int,
    dmd_lower: int,
    dmd_upper: int,
    cap_lower: int,
    cap_upper: int,
) -> dict:
    """
    Generates a CVRP problem by sampling customers from the mesh.
    """
    if dmd_lower < 1:
        raise ValueError("dmd_lower must be at least 1")
    if cap_lower < 1:
        raise ValueError("cap_lower must be at least 1")
    if dmd_upper < dmd_lower:
        raise ValueError("dmd_upper must be greater than or equal to dmd_lower")
    if cap_upper < cap_lower:
        raise ValueError("cap_upper must be greater than or equal to cap_lower")
    if dmd_upper > cap_lower:
        raise ValueError("dmd_upper must be less than or equal to cap_lower")
    if problem_size > city_size:
        raise ValueError("problem_size cannot be larger than city_size")

    cust_indices = np.random.choice(city_size, problem_size, replace=False) + 1
    cust_demand = np.random.randint(dmd_lower, dmd_upper + 1, problem_size)

    problem = np.concatenate([[0], cust_indices])  # Add depot to index
    demand = np.concatenate([[0], cust_demand])  # The depot has no demand
    capacity = np.random.randint(cap_lower, cap_upper + 1)  # This is an int

    problem = problem.astype(np.int32)
    demand = demand.astype(np.int32)

    return {"problem": problem, "demand": demand, "capacity": capacity}


def get_solution(mesh_city: dict, problem: dict, max_runtime: int) -> dict:
    """Solves the CVRP using PyVRP with precomputed geodesic distances."""
    problem_indices = problem["problem"]

    # Get the precomputed geodesic matrix
    geodesic_matrix = mesh_city["geodesic_matrix"]

    # Map problem indices to city indices for the distance matrix lookup
    problem_to_city_map = {p_idx: c_idx for c_idx, p_idx in enumerate(problem_indices)}

    # Function to get geodesic distance between two points using the precomputed matrix
    def get_distance(i, j):
        city_i = problem_to_city_map[i]
        city_j = problem_to_city_map[j]
        return geodesic_matrix[city_i, city_j]

    # Set up PyVRP model with custom distance function
    m = Model()
    m.add_vehicle_type(1000, capacity=problem["capacity"])

    # Add locations
    COORDS = mesh_city["city"][problem_indices]
    COORDS = (COORDS[:, :2] * 1000).round().astype(np.int32).tolist()
    DEMANDS = problem["demand"].tolist()

    depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1])
    clients = [
        m.add_client(x=COORDS[idx][0], y=COORDS[idx][1], delivery=DEMANDS[idx])
        for idx in range(1, len(COORDS))
    ]

    # Add edges with precomputed geodesic distances
    locations = [depot] + clients
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            if i != j:
                distance = get_distance(problem_indices[i], problem_indices[j])
                m.add_edge(frm, to, distance=int(distance * 1000))

    # Solve and return solution
    res = m.solve(stop=MaxRuntime(max_runtime), display=False)
    res_dict = parse_pyvrp_solution(res.best)

    # Convert distance back to float
    res_dict["distance"] = res_dict["distance"] / 1000

    # Convert solution back into the city space
    # solution = [problem["problem"][idx] for idx in res_dict["solution"]]
    solution = res_dict["solution"]

    # Add node_flag format
    res_dict["node_flag"] = transform_solution(solution)

    return res_dict


def generate_and_solve_mesh_problem(args):
    (
        mesh_city,
        city_size,
        problem_size,
        dmd_lower,
        dmd_upper,
        cap_lower,
        cap_upper,
        max_runtime,
    ) = args

    # Re-seed the RNG in each process
    np.random.seed(None)

    problem = get_problem(
        city_size, problem_size, dmd_lower, dmd_upper, cap_lower, cap_upper
    )
    solution = get_solution(mesh_city, problem, max_runtime)

    return {
        "problem": problem["problem"],
        "demand": problem["demand"],
        "capacity": problem["capacity"],
        "distance": solution["distance"],
        "node_flag": solution["node_flag"],
    }


def produce_problem_instances(
    mesh_path: str,
    num_problems: int,
    problem_size: int,
    filename: str,
    dmd_lower: int = 1,
    dmd_upper: int = 10,
    cap_lower: int = 20,
    cap_upper: int = 50,
    max_runtime: int = 5,
    num_customers: int = 100,
) -> None:
    """
    Generates multiple mesh-based CVRP problem-solution pairs and saves them in an HDF5 file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Generate the mesh city once
    logging.info(f"Generating mesh city with {num_customers} customers...")
    mesh_city, city_size = get_mesh_city(mesh_path, num_customers)

    # Save the mesh data
    logging.info(f"Saving mesh data to {filename}...")
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("vertices", data=mesh_city["vertices"])
        hf.create_dataset("faces", data=mesh_city["faces"])
        hf.create_dataset("city", data=mesh_city["city"])
        hf.create_dataset("city_indices", data=mesh_city["city_indices"])
        hf.create_dataset("geodesic_matrix", data=mesh_city["geodesic_matrix"])

    # Create a list of arguments for each problem
    args_list = [
        (
            mesh_city,
            city_size,
            problem_size,
            dmd_lower,
            dmd_upper,
            cap_lower,
            cap_upper,
            max_runtime,
        )
    ] * num_problems

    # Generate and solve problems in parallel using context manager
    logging.info(f"Generating and solving {num_problems} problems...")
    start_time = time.time()

    # Using multiprocessing context manager
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(
                    generate_and_solve_mesh_problem,
                    args_list,
                    chunksize=max(1, num_problems // os.cpu_count()),
                ),
                total=num_problems,
                desc="Solving problem instances",
            )
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(
        f"Time taken to generate {num_problems} problems: {elapsed_time/60:.2f}m"
    )

    # Save problem-solution data to HDF5 file
    logging.info("Saving problem...")
    with h5py.File(filename, "a") as hf:
        hf.create_dataset(
            "problems",
            data=np.array([result["problem"] for result in results], dtype=np.int32),
        )
        hf.create_dataset(
            "demands",
            data=np.array([result["demand"] for result in results], dtype=np.float32),
        )
        hf.create_dataset(
            "capacities",
            data=np.array([result["capacity"] for result in results], dtype=np.int32),
        )
        hf.create_dataset(
            "distances",
            data=np.array([result["distance"] for result in results], dtype=np.float32),
        )
        hf.create_dataset(
            "node_flags",
            data=np.array([result["node_flag"] for result in results], dtype=np.int32),
        )

    logging.info(f"Successfully saved {num_problems} problems to {filename}")


def access_mesh_cvrp_data(filename: str, problem_index: int = 0) -> dict:
    """
    Accesses mesh-based CVRP problem-solution data from an HDF5 file.
    """
    with h5py.File(filename, "r") as hf:
        # Mesh data
        vertices = hf["vertices"][:]
        faces = hf["faces"][:]
        city = hf["city"][:]
        city_indices = hf["city_indices"][:]
        geodesic_matrix = hf["geodesic_matrix"][:]

        # Problem-solution data
        problem = hf["problems"][problem_index]
        demand = hf["demands"][problem_index]
        capacity = hf["capacities"][problem_index]
        distance = hf["distances"][problem_index]
        node_flag = hf["node_flags"][problem_index]

    return {
        "vertices": vertices,
        "faces": faces,
        "city": city,
        "city_indices": city_indices,
        "geodesic_matrix": geodesic_matrix,
        "problem": problem,
        "demand": demand,
        "capacity": capacity,
        "distance": distance,
        "node_flag": node_flag,  # Added node_flag
    }


if __name__ == "__main__":
    # TODO: Sync this mesh path with the actual path in the architecture/ config file
    mesh_path = "main_model/disk/meshes/sphere.obj"
    filename_train = "main_model/disk/problems/mesh_cvrp_data_train.h5"
    filename_test = "main_model/disk/problems/mesh_cvrp_data_test.h5"
    num_problems_train = 100
    num_problems_test = 100
    problem_size = 100
    num_customers = 100

    # TODO: need to hardcode the depot location

    # Generate training problems
    produce_problem_instances(
        mesh_path=mesh_path,
        num_problems=num_problems_train,
        problem_size=problem_size,
        filename=filename_train,
        dmd_lower=1,
        dmd_upper=9,
        cap_lower=50,
        cap_upper=50,
        max_runtime=5,
        num_customers=num_customers,  # Sample all customers
    )

    # Generate testing problems
    produce_problem_instances(
        mesh_path=mesh_path,
        num_problems=num_problems_test,
        problem_size=problem_size,
        filename=filename_test,
        dmd_lower=1,
        dmd_upper=9,
        cap_lower=50,
        cap_upper=50,
        max_runtime=5,
        num_customers=num_customers,  # Sample all customers
    )

    # Access and visualize a problem
    data = access_mesh_cvrp_data(filename_train, 0)
    print(f"Problem size: {len(data['problem'])}")
    print(f"Solution distance: {data['distance']}")
    print(f"Node-flag format: {data['node_flag']}")
