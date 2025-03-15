import os
import multiprocessing as mp
import random
import numpy as np
import trimesh
from tqdm import tqdm
import logging

from pygeodesic import geodesic

from main_model.src.utils.config import load_config, parse_args

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Send logs to the console (stdout)
        # logging.FileHandler("my_log_file.log"),  # Optionally, write logs to a file
    ],
)


def visualize_ssad(vertices: np.ndarray, triangles: np.ndarray, source_index: int):
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = None
    distances, _ = geoalg.geodesicDistances(source_indices, target_indices)
    return distances


def visualize_two_pts(
    vertices: np.ndarray, triangles: np.ndarray, source_index: int, dest_index: int
):
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = np.array([dest_index])
    distances, _ = geoalg.geodesicDistance(source_indices, target_indices)
    return distances


def data_prepare_gen_dataset(
    object_file: str,
    output_path: str,
    num_sources: int,
    num_destinations: int,
    tqdm_on: bool = True,
):
    """
    Prepares and saves the distances between the sources and destinations for the given object file.

    Args:
        object_file: str, path to the object file
        output_path: str, path to save the output file
        num_sources: int, number of sources to sample
        num_destinations: int, number of destinations per source
        tqdm_on: bool, whether to display the progress bar
    """

    vertices = []
    triangles = []

    with open(object_file, "r") as f:
        lines = f.readlines()
        for each in lines:
            if each.startswith("v "):
                temp = each.split()
                vertices.append([float(temp[1]), float(temp[2]), float(temp[3])])
            if each.startswith("f "):
                temp = each.split()
                #
                temp[3] = temp[3].split("/")[0]
                temp[1] = temp[1].split("/")[0]
                temp[2] = temp[2].split("/")[0]
                triangles.append([int(temp[1]) - 1, int(temp[2]) - 1, int(temp[3]) - 1])
    vertices = np.array(vertices)
    triangles = np.array(triangles)

    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    result = np.array([[0, 0, 0]])

    sources = np.random.randint(low=0, high=len(vertices), size=[num_sources])

    # only the process on process #0 will be displayed
    # this should not be problematic or confusing on most homogeneous CPUs
    progress_iterator = tqdm(range(num_sources)) if tqdm_on else range(num_sources)
    for i in progress_iterator:
        source_indices = np.array([sources[i]])
        target_indices = np.random.randint(
            low=0, high=len(vertices), size=[num_destinations]
        )
        if (source_indices.max() >= len(vertices)) or (
            target_indices.max() >= len(vertices)
        ):
            raise ValueError("Index out of range")

        distances, _ = geoalg.geodesicDistances(source_indices, target_indices)

        sources_sampled = source_indices.repeat([num_destinations]).reshape([-1, 1])
        targets_sampled = target_indices.reshape([-1, 1])
        distances_sampled = distances.reshape([-1, 1])
        new = np.concatenate([sources_sampled, targets_sampled, distances_sampled], -1)
        result = np.concatenate([result, new])

    np.savetxt(output_path, result)


def computation_thread(
    filename: str,
    object_name: str,
    num_sources: int,
    num_targets: int,
    idx: int | None = None,
):
    """
    Runs the computation thread for the given object file.

    Args:
        filename: str, path to the object file
        object_name: str, name of the object
        num_sources: int, number of sources to sample
        num_targets: int, number of targets per source
        idx: int, index of the object
    """
    assert idx is not None, "An index ('idx') has to be given"

    tqdm_on = False
    if idx == 0:
        tqdm_on = True

    logging.info(filename + object_name)
    output = object_name + "_" + str(idx)

    data_prepare_gen_dataset(
        filename, output, num_sources, num_targets, tqdm_on=tqdm_on
    )


if __name__ == "__main__":
    """
    Generates the npz files and filelists for the mesh data. Samples the geodesic distances for both training and test datasets.

    Args:
        PATH_TO_MESH: str, path to the mesh files
        PATH_TO_OUTPUT_NPZ: str, path to the output npz files
        PATH_TO_OUTPUT_FILELIST: str, path to the output filelist

        split_ratio: float, ratio of training data

        num_train_sources: int, training set: number of sources to sample
        num_train_targets_per_source: int, training set: destinations per source
        num_test_sources: int, testing set: number of sources to sample
        num_test_targets_per_source: int, testing set: destinations per source

        file_size_threshold: int, file size threshold
        threads: int, number of threads. 0 uses all cores
    """
    # Load the config file
    args = parse_args()
    config = load_config(args.config)

    # Initialize paths
    PATH_TO_MESH = config["data"]["preparation"]["path_to_mesh"]
    PATH_TO_OUTPUT_NPZ = config["data"]["preparation"]["path_to_output_npz"]
    PATH_TO_OUTPUT_FILELIST = config["data"]["preparation"]["path_to_output_filelist"]

    # Define the parameters
    SPLIT_RATIO = config["data"]["preparation"]["split_ratio"]
    FILE_SIZE_THRESHOLD = config["data"]["preparation"]["file_size_threshold"]
    LARGE_DISTANCE_THRESHOLD = 1e8

    # number of sources and targets for sampling the distances
    num_train_sources = config["data"]["preparation"]["num_train_sources"]
    num_train_targets_per_source = config["data"]["preparation"][
        "num_train_targets_per_source"
    ]
    num_test_sources = config["data"]["preparation"]["num_test_sources"]
    num_test_targets_per_source = config["data"]["preparation"][
        "num_test_targets_per_source"
    ]

    threads = config["data"]["preparation"]["threads"]
    object_name = None

    assert threads >= 0 and type(threads) == int

    if threads == 0:
        threads = mp.cpu_count()
        logging.info(f"Automatically utilize all CPU cores ({threads})")
    else:
        logging.info(f"{threads} CPU cores are utilized!")

    # make dirs, if not exist
    if not os.path.exists(PATH_TO_OUTPUT_NPZ):
        os.mkdir(PATH_TO_OUTPUT_NPZ)
    if not os.path.exists(PATH_TO_OUTPUT_FILELIST):
        os.mkdir(PATH_TO_OUTPUT_FILELIST)

    all_files = []
    for mesh in os.listdir(PATH_TO_MESH):
        # check if the file is too large
        if os.path.getsize(PATH_TO_MESH + mesh) < FILE_SIZE_THRESHOLD:
            all_files.append(os.path.join(PATH_TO_MESH, mesh))

    object_names = all_files
    for i in range(len(object_names)):
        if object_names[i].endswith(".obj"):
            object_names[i] = object_names[i][:-4]

    random.shuffle(object_names)
    train_num = int(len(object_names) * SPLIT_RATIO)
    test_num = len(object_names) - train_num
    train_objects = object_names[:train_num]
    test_objects = object_names[train_num:]

    # handle the training set
    for i in tqdm(range(len(train_objects)), "Processing the training set"):
        object_name = train_objects[i]
        if object_name.split("/")[-1][0] == ".":
            raise ValueError("Not an obj file")

        filename_out = PATH_TO_OUTPUT_NPZ + object_name + ".npz"
        if os.path.exists(filename_out):
            print(f"File already exists, skipping {object_name}...")
            continue

        filename = object_name + ".obj"

        thread_list = []

        pool = []

        for t in range(threads):
            task = mp.Process(
                target=computation_thread,
                args=(
                    filename,
                    object_name,
                    num_train_sources // threads,
                    num_train_targets_per_source,
                    t,
                ),
            )
            task.start()
            pool.append(task)
        for t, task in enumerate(pool):
            task.join()
            thread_list.append(object_name + "_" + str(t))

        try:
            for i in range(len(thread_list)):
                # train data
                with open(object_name + "_" + str(i), "r") as f:
                    data = f.read()
                with open(object_name, "a") as f:
                    f.write(data)
        except Exception as e:
            logging.info("error:" + str(e))

        for each in thread_list:
            os.remove(each)

        filename_in = object_name + ".obj"
        dist_in = object_name
        filename_out = PATH_TO_OUTPUT_NPZ + object_name.split("/")[-1] + ".npz"
        try:
            mesh = trimesh.load_mesh(filename_in)
            dist = np.loadtxt(dist_in)
        except Exception as e:
            logging.info(f"load {filename_in} or {dist_in} failed with error: {e}")

        # delete the dist_in
        os.remove(dist_in)

        unique_edges = mesh.edges_unique
        unique_edges_rev = np.concatenate(
            [unique_edges[:, 1:], unique_edges[:, :1]], 1
        )  # reverse edges
        edges = np.concatenate(
            [unique_edges, unique_edges_rev]
        )  # add normal and reversed edge pairs for "undirectional" edges

        # sanity check
        vertices = mesh.vertices
        if dist.max() > LARGE_DISTANCE_THRESHOLD:
            raise ValueError("Distance too large!")
        elif (dist.astype(np.float32).max()) >= vertices.shape[0]:
            raise ValueError("Encountered a trimesh loading error!")

        np.savez(
            filename_out,
            edges=edges,
            vertices=mesh.vertices.astype(np.float32),
            normals=mesh.vertex_normals.astype(np.float32),
            faces=mesh.faces.astype(np.float32),
            dist_val=dist[:, 2:].astype(np.float32),
            dist_idx=dist[:, :2].astype(np.uint16),
        )

    # handle the case for the training set
    for i in tqdm(range(len(test_objects)), "Processing the testing set"):
        object_name = test_objects[i]
        if object_name.split("/")[-1][0] == ".":
            continue  # not an obj file

        filename_out = PATH_TO_OUTPUT_NPZ + object_name + ".npz"
        if os.path.exists(filename_out):
            continue

        filename = object_name + ".obj"

        thread_list = []

        pool = []

        for t in range(threads):
            task = mp.Process(
                target=computation_thread,
                args=(
                    filename,
                    object_name,
                    num_test_sources // threads,
                    num_test_targets_per_source,
                    t,
                ),
            )
            task.start()
            pool.append(task)
        for t, task in enumerate(pool):
            task.join()
            thread_list.append(object_name + "_" + str(t))

        try:
            for i in range(len(thread_list)):
                with open(object_name + "_" + str(i), "r") as f:
                    data = f.read()
                with open(object_name, "a") as f:
                    f.write(data)
        except Exception as e:
            logging.info("Error:" + str(e))

        for each in thread_list:
            os.remove(each)

        filename_in = object_name + ".obj"
        dist_in = object_name
        filename_out = PATH_TO_OUTPUT_NPZ + object_name.split("/")[-1] + ".npz"
        try:
            mesh = trimesh.load_mesh(filename_in)
            dist = np.loadtxt(dist_in)
        except Exception as e:
            logging.info(f"load {filename_in} or {dist_in} failed with error: {e}")

        # delete the dist_in
        os.remove(dist_in)

        aa = mesh.edges_unique
        bb = np.concatenate([aa[:, 1:], aa[:, :1]], 1)
        cc = np.concatenate([aa, bb])

        # sanity check
        vertices = mesh.vertices
        if dist.max() > LARGE_DISTANCE_THRESHOLD:
            raise ValueError("Distance too large!")
        elif (dist.astype(np.float32).max()) >= vertices.shape[0]:
            raise ValueError("Encountered a trimesh loading error!")

        np.savez(
            filename_out,
            edges=cc,
            vertices=mesh.vertices.astype(np.float32),
            normals=mesh.vertex_normals.astype(np.float32),
            faces=mesh.faces.astype(np.float32),
            dist_val=dist[:, 2:].astype(np.float32),
            dist_idx=dist[:, :2].astype(np.uint16),
        )

    logging.info("\nThe npz data generation finished. Now generating filelist...\n")

    train_lines = []
    for each in tqdm(train_objects):
        filename_out = PATH_TO_OUTPUT_NPZ + each.split("/")[-1] + ".npz"
        try:
            dist = np.load(filename_out)

            # sanity check
            if (
                dist["dist_val"].max() != np.inf
                and dist["dist_val"].max() < LARGE_DISTANCE_THRESHOLD
            ):
                train_lines.append(filename_out + "\n")
            else:
                raise ValueError("File contains inf for the distances!")
        except Exception as e:
            raise ValueError("load " + filename_out + " failed with error: " + str(e))

    test_lines = []
    for each in tqdm(test_objects):
        filename_out = PATH_TO_OUTPUT_NPZ + each.split("/")[-1] + ".npz"
        try:
            dist = np.load(filename_out)

            # sanity check
            if (
                dist["dist_val"].max() != np.inf
                and dist["dist_val"].max() < LARGE_DISTANCE_THRESHOLD
            ):
                test_lines.append(filename_out + "\n")
            else:
                raise ValueError("File contains inf for the distances!")
        except Exception as e:
            raise ValueError("load " + filename_out + " failed with error: " + str(e))

    with open(PATH_TO_OUTPUT_FILELIST + "filelist_train.txt", "w") as f:
        f.writelines(train_lines)

    with open(PATH_TO_OUTPUT_FILELIST + "filelist_test.txt", "w") as f:
        f.writelines(test_lines)

    logging.info("The filelist generation finished.")
