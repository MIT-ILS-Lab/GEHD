"""
This file contains the dataloader for the decoder. Part of the code is adapted from the LEHD paper
titled "Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization" by Fu Luo et al (https://openreview.net/forum?id=RBI4oAbdpm).
Their GitHub repository can be found at https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD.
"""

import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Union, Optional, Any, Generator, Iterator

import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)


def process_problem(
    hf_path: str, i: int
) -> Tuple[List[float], int, List[float], float, List[List[int]]]:
    """Top-level function to process a single problem."""
    with h5py.File(hf_path, "r") as hf:
        problem = hf["problems"][i]
        demand = hf["demands"][i]
        capacity = hf["capacities"][i]
        distance = hf["distances"][i]
        node_flag = tow_col_nodeflag(hf["node_flags"][i])
    return problem.tolist(), int(capacity), demand.tolist(), float(distance), node_flag


def tow_col_nodeflag(node_flag: np.ndarray) -> List[List[int]]:
    """Convert one-row node_flag to two-column format."""
    V = int(len(node_flag) / 2)
    return [[int(node_flag[i]), int(node_flag[V + i])] for i in range(V)]


def reformat_solution(solution: torch.Tensor, problem: torch.Tensor) -> torch.Tensor:
    """
    Adapts the batch processing for no batching (single sample).
    Args:
        solution: The solution to be reformatted.
        problem: The problem to be reformatted.
    Returns:
        The reformatted solution.
    """

    # extend the solution array with 0s on the first place
    reindex_list = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int),
            solution[:, 0],
            torch.zeros((1,), dtype=torch.int),
        ),
        dim=0,
    )

    reindex_flag = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int),
            solution[:, 1],
            torch.zeros((1,), dtype=torch.int),
        ),
        dim=0,
    )

    # orders the problem nodes by the depot and solution indices for future processing
    solution_extended = problem[reindex_list]
    # concatenate the flag to the reindexed problems
    solution_extended = torch.cat((solution_extended, reindex_flag.unsqueeze(1)), dim=1)
    return solution_extended


class MultiInstanceGEHDDataset(Dataset):
    """Wrapper dataset that handles multiple instance sizes"""

    def __init__(
        self,
        data_paths: Dict[int, str],
        mode: str = "test",
        episodes: int = -1,
        distort: bool = False,
    ) -> None:
        self.datasets = {
            size: GEHDDataset(path, mode, episodes, distort)
            for size, path in data_paths.items()
        }
        self.instance_sizes = list(data_paths.keys())
        self.current_size = None  # Track current active size

    def set_size(self, size: int) -> None:
        """Switch active instance size"""
        if size not in self.datasets:
            raise ValueError(f"Invalid instance size {size}")
        self.current_size = size
        self.active_dataset = self.datasets[size]

    def __len__(self) -> int:
        return len(self.active_dataset) if self.current_size else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.active_dataset[idx]

    def get_all_sizes(self) -> List[int]:
        return self.instance_sizes


class GEHDBatchSampler(torch.utils.data.Sampler):
    """
    BatchSampler that groups items into batches where each batch uses a fixed subpath length.
    This ensures all items in a batch have the same sequence length.
    """

    def __init__(
        self,
        dataset_size: int,
        problem_size: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> None:
        self.dataset_size = dataset_size
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_batches = dataset_size // batch_size
        if not drop_last and dataset_size % batch_size != 0:
            self.num_batches += 1
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        # Create random indices
        if self.shuffle:
            indices = torch.randperm(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))

        # Yield batches
        for i in range(0, self.dataset_size, self.batch_size):
            batch_indices = indices[i : min(i + self.batch_size, self.dataset_size)]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # fixed_length = torch.randint(low=4, high=self.problem_size + 1, size=[1])[0]
            # batch_indices = [(x, fixed_length.item()) for x in batch_indices]
            batch_indices = [(x, 0) for x in batch_indices]  # don't cut solutions

            yield batch_indices


class InfiniteGEHDBatchSampler:
    """
    InfiniteBatchSampler that groups items into batches where each batch uses a fixed subpath length.
    This ensures all items in a batch have the same sequence length.
    """

    def __init__(self, batch_sampler: GEHDBatchSampler) -> None:
        self.batch_sampler = batch_sampler

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        while True:
            # Create a fresh iterator from the batch sampler each time
            batch_iter = iter(self.batch_sampler)
            for batch in batch_iter:
                yield batch

    def __len__(self) -> int:
        # Return the length of the underlying batch sampler
        return len(self.batch_sampler)


class GEHDDataset(Dataset):
    """
    Dataset class for the GEHD problem. Parts of the class are adapted from the LEHD paper.
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        episodes: int = 100,
        distort: bool = False,
    ) -> None:
        """
        Initialize the GEHDDataset.
        Args:
            data_path: The path to the data file.
            mode: The mode of the dataset.
            episodes: The number of episodes to load.
            distort: Whether to use path distortion.
        """
        self.data_path = data_path
        self.mode = mode
        self.episodes = episodes
        self.distort = distort

        # Load the raw data
        self.load_raw_data(self.episodes)

    def __len__(self) -> int:
        return len(self.raw_data_problems)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        # Get a single problem instance
        if isinstance(idx, int):
            idx = (idx, 0)  # Default fixed_length if not provided
        idx, fixed_length = idx

        # Get problem indices
        problems = self.raw_data_problems[idx]
        capacity = self.raw_data_capacity[idx]
        demand = self.raw_data_demand[idx]
        solution = self.raw_data_node_flag[idx]
        cost = self.raw_data_cost[idx]

        # Create the problem representation
        capacity_expanded = capacity.unsqueeze(0).repeat(solution.shape[0] + 1)
        problem = torch.cat(
            (
                problems.unsqueeze(-1),
                problems.unsqueeze(-1),  # TODO: Delete this
                demand.unsqueeze(-1),
                capacity_expanded.unsqueeze(-1),
            ),
            dim=1,
        )

        # Apply subpath sampling if needed, using the provided fixed_length
        if self.distort:
            problem, solution = self.sampling_subpaths(problem, solution, fixed_length)

        solution = reformat_solution(solution, problem)

        return {
            "solution": solution,
            "capacity": capacity,
            "cost": cost,
        }

    def load_raw_data(self, episode: int = 1000000) -> None:
        logging.info(f"Start loading {self.mode} dataset from HDF5 file...")

        assert self.data_path.endswith(".h5"), "Data file must be in HDF5 format"

        with h5py.File(self.data_path, "r") as hf:
            total_problems = len(hf["problems"])
            num_problems = (
                total_problems if episode == -1 else min(episode, total_problems)
            )

        logging.info(f"Loading {num_problems} problems from {self.data_path}...")

        # Use multiprocessing to load data in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        partial(process_problem, self.data_path), range(num_problems)
                    ),
                    total=num_problems,
                    desc=f"Loading {self.mode} data",
                )
            )

        # Unpack results into separate lists
        (
            self.raw_data_problems,
            self.raw_data_capacity,
            self.raw_data_demand,
            self.raw_data_cost,
            self.raw_data_node_flag,
        ) = zip(*results)

        # Convert lists to tensors
        self.raw_data_problems = torch.tensor(
            self.raw_data_problems, requires_grad=False
        )
        self.raw_data_capacity = torch.tensor(
            self.raw_data_capacity, requires_grad=False
        )
        self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False)
        self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False)
        self.raw_data_node_flag = torch.tensor(
            self.raw_data_node_flag, requires_grad=False
        )

        logging.info(
            f"Loading {self.mode} dataset done! Loaded {len(self.raw_data_capacity)} problems."
        )

    def shuffle_data(self) -> None:
        # Shuffle the training set data
        index = torch.randperm(len(self.raw_data_problems)).long()
        self.raw_data_problems = self.raw_data_problems[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]

    def vrp_whole_and_solution_subrandom_inverse(
        self, solution: torch.Tensor
    ) -> torch.Tensor:
        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])
            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1. Find the number of subtours in each instance
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat(
            (solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1
        )
        start_from_depot = fake_solution.nonzero()
        start_from_depot_1 = start_from_depot[:, 1]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)
        sub_tours_length = start_from_depot_2 - start_from_depot_1
        max_subtour_length = torch.max(sub_tours_length)

        # 2. Extract and process subtours
        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(
            visit_depot_num, dim=0
        )
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = (
            torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(
                len(repeat_solutions_node), 1
            )
            >= start_from_depot2[:, 1][:, None]
        )
        x2 = (
            torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(
                len(repeat_solutions_node), 1
            )
            <= start_from_depot3[:, 1][:, None]
        )
        x3 = (x1 * x2).long()
        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(
            len(repeat_solutions_node), 1
        ) < (start_from_depot2[:, 1][:, None] + max_subtour_length)
        x5 = x1 * x4
        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 3. Randomly flip subtours
        clockwise_or_not = torch.rand(len(sub_tours_padding))
        clockwise_or_not_bool = clockwise_or_not.le(0.5)
        sub_tours_padding[clockwise_or_not_bool] = torch.flip(
            sub_tours_padding[clockwise_or_not_bool], dims=[1]
        )

        # 4. Map back to original solution
        sub_tourss_back = sub_tourss
        sub_tourss_back[x5] = sub_tours_padding.ravel()
        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(
            batch_size, problem_size
        )
        solution_flip = torch.cat(
            (solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2
        )

        return solution_flip

    def vrp_whole_and_solution_subrandom_shift_V2inverse(
        self, solution: torch.Tensor
    ) -> torch.Tensor:
        """
        For each instance, shift randomly so that different end_with depot nodes can reach the last digit.
        """
        problem_size = solution.shape[1]
        batch_size = solution.shape[0]

        start_from_depot = solution[:, :, 1].nonzero()
        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1
        end_with_depot[:, 1] = torch.roll(end_with_depot[:, 1], dims=0, shifts=-1)
        visit_depot_num = solution[:, :, 1].sum(1)
        min_length = torch.min(visit_depot_num)

        first_node_index = torch.randint(low=0, high=min_length, size=[1])[
            0
        ]  # in [0,N)

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long()

        pick_end_with_depot_index = temp_index_torch + first_node_index
        pick_end_with_depot_ = end_with_depot[pick_end_with_depot_index][:, 1]
        first_index = pick_end_with_depot_
        end_indeex = pick_end_with_depot_ + problem_size

        index = torch.arange(2 * problem_size)[None, :].repeat(batch_size, 1)
        x1 = index > first_index[:, None]
        x2 = index <= end_indeex[:, None]
        x3 = x1.int() * x2.int()
        double_solution = solution.repeat(1, 2, 1)
        solution = double_solution[x3.gt(0.5)[:, :, None].repeat(1, 1, 2)].reshape(
            batch_size, problem_size, 2
        )

        return solution

    def sampling_subpaths(
        self,
        problem: torch.Tensor,
        solution: torch.Tensor,
        fixed_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # problem shape (V+1,4)
        # solution shape (V,2)

        # 1. Extract subtour
        problems_size = problem.shape[0] - 1  # Excluding depot
        embedding_size = problem.shape[1]

        # Use fixed length if provided, otherwise random
        if fixed_length is not None and fixed_length > 0:
            length_of_subpath = fixed_length
        else:
            # Random length of subpath between 4 and problem size
            length_of_subpath = torch.tensor(problems_size)

        # Apply data augmentation
        solution = self.vrp_whole_and_solution_subrandom_inverse(
            solution.unsqueeze(0)
        ).squeeze(0)
        solution = self.vrp_whole_and_solution_subrandom_shift_V2inverse(
            solution.unsqueeze(0)
        ).squeeze(0)

        # Find points that start from depot
        start_from_depot = solution[:, 1].nonzero().squeeze(1)
        end_with_depot = start_from_depot.clone()
        end_with_depot = end_with_depot - 1
        if end_with_depot[0] < 0:
            end_with_depot[0] = solution.shape[0] - 1

        # Count depot visits
        visit_depot_num = torch.sum(solution[:, 1])

        # Randomly select end point
        p = torch.rand(1)
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(
            select_end_with_depot_node_index
        ).long()

        # This is the point at which the instance is randomly selected with an end with depot
        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index]

        # Create double solution for circular handling
        double_solution = torch.cat((solution, solution), dim=0)
        select_end_with_depot_node = select_end_with_depot_node + problems_size

        # Create indices for subpath extraction
        offset = select_end_with_depot_node - length_of_subpath + 1
        indexx = torch.arange(length_of_subpath) + offset
        sub_tour = double_solution[indexx, :]

        # Calculate the capacity of the first point
        start_index = indexx[0]

        # Process capacity constraints
        x1 = torch.arange(solution.shape[0]) <= start_index
        before_is_via_depot_all = solution[:, 1] * x1
        visit_depot_num_2 = torch.sum(before_is_via_depot_all)

        # Update node indices for the subpath
        sub_solution_node = sub_tour[:, 0]
        new_sulution_ascending, rank = torch.sort(
            sub_solution_node, dim=-1, descending=False
        )
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        sub_tour[:, 0] = new_sulution_rank + 1

        # Create the new problem representation
        index_2, _ = (
            torch.cat(
                (
                    new_sulution_ascending,
                    new_sulution_ascending,
                    new_sulution_ascending,
                    new_sulution_ascending,
                )
            )
            .type(torch.long)
            .sort(dim=-1, descending=False)
        )

        index_3 = torch.arange(embedding_size, dtype=torch.long)
        index_3 = index_3.repeat(length_of_subpath)

        new_data = problem[index_2, index_3].view(length_of_subpath, embedding_size)
        new_data = torch.cat((problem[0, :].unsqueeze(dim=0), new_data), dim=0)

        return new_data, sub_tour


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    solutions = torch.stack([item["solution"] for item in batch])
    capacities = torch.stack([item["capacity"] for item in batch])
    costs = torch.stack([item["cost"] for item in batch])

    batch_dict = {
        "solutions": solutions,
        "capacities": capacities.float(),
        "costs": costs.float(),
    }

    return batch_dict


def get_dataset(
    config: Dict[str, Any],
) -> Tuple[Union[GEHDDataset, MultiInstanceGEHDDataset], Any]:
    # Create dataset
    if config["mode"] == "train":
        dataset = GEHDDataset(
            data_path=config["env"]["data_path"],
            episodes=config["episodes"],
            mode=config["mode"],
            distort=config["env"]["distort"],
        )
    elif config["mode"] == "test":
        dataset = MultiInstanceGEHDDataset(
            data_paths=config["env"]["data_path"],
            episodes=config["episodes"],
            mode=config["mode"],
            distort=config["env"]["distort"],
        )
    else:
        raise ValueError(f"Invalid mode: {config['mode']}. Must be 'train' or 'test'.")

    return dataset, collate_batch
