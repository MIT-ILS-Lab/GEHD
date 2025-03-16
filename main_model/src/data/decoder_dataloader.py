import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class LEHDBatchSampler(torch.utils.data.Sampler):
    """
    BatchSampler that groups items into batches where each batch uses a fixed subpath length.
    This ensures all items in a batch have the same sequence length.
    """

    def __init__(self, dataset_size, problem_size, batch_size, drop_last=False):
        self.dataset_size = dataset_size
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_batches = dataset_size // batch_size
        if not drop_last and dataset_size % batch_size != 0:
            self.num_batches += 1

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Create random indices
        indices = torch.randperm(self.dataset_size).tolist()

        # Yield batches
        for i in range(0, self.dataset_size, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            fixed_length = torch.randint(low=4, high=self.problem_size + 1, size=[1])[0]
            batch_indices = [(x, fixed_length.item()) for x in batch_indices]
            yield batch_indices


class InfiniteLEHDBatchSampler:
    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler

    def __iter__(self):
        while True:
            # Create a fresh iterator from the batch sampler each time
            batch_iter = iter(self.batch_sampler)
            for batch in batch_iter:
                yield batch

    def __len__(self):
        # Return the length of the underlying batch sampler
        return len(self.batch_sampler)


class LEHDDataset(Dataset):
    def __init__(
        self, data_path, mode="train", episodes=100, sub_path=False, device=None
    ):
        self.data_path = data_path
        self.mode = mode
        self.episodes = episodes
        self.sub_path = sub_path
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load the raw data
        self.load_raw_data(self.episodes)

        # Load mesh data for computing node locations
        self.load_mesh_data()

    def __len__(self):
        return len(self.raw_data_problems)

    def __getitem__(self, args):
        # Get a single problem instance
        idx = args[0]
        fixed_length = args[1]

        # Get problem indices
        problem_indices = self.raw_data_problems[idx]

        # Get node coordinates from mesh city using problem indices
        # nodes = self.city[problem_indices]

        capacity = self.raw_data_capacity[idx]
        demand = self.raw_data_demand[idx]
        solution = self.raw_data_node_flag[idx]

        # Create the problem representation
        capacity_expanded = capacity.unsqueeze(0).repeat(solution.shape[0] + 1)
        problem = torch.cat(
            (
                problem_indices.unsqueeze(-1),
                demand.unsqueeze(-1),
                capacity_expanded.unsqueeze(-1),
            ),
            dim=1,
        )

        # Apply subpath sampling if needed, using the provided fixed_length
        if self.sub_path:
            problem, solution = self.sampling_subpaths(
                problem, solution, fixed_length=fixed_length
            )

        return {"problem": problem, "solution": solution, "capacity": capacity}

    def load_mesh_data(self):
        """Load the mesh data to get node coordinates"""
        with h5py.File(self.data_path, "r") as hf:
            # Load mesh data
            self.vertices = torch.tensor(hf["vertices"][:], requires_grad=False).to(
                self.device
            )
            self.faces = torch.tensor(hf["faces"][:], requires_grad=False).to(
                self.device
            )
            self.city = torch.tensor(hf["city"][:], requires_grad=False).to(self.device)
            self.city_indices = torch.tensor(
                hf["city_indices"][:], requires_grad=False
            ).to(self.device)
            self.geodesic_matrix = torch.tensor(
                hf["geodesic_matrix"][:], requires_grad=False
            ).to(self.device)

        logging.info(f"Loaded mesh data with {len(self.city)} city points")

    def load_raw_data(self, episode=1000000):
        logging.info(f"Start loading {self.mode} dataset from HDF5 file...")

        # Helper function to convert one-row node_flag to two-column format
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([int(node_flag[i]), int(node_flag[V + i])])
            return tow_col_node_flag

        assert self.data_path.endswith(".h5"), "Data file must be in HDF5 format"

        with h5py.File(self.data_path, "r") as hf:
            # Determine how many problems to load
            total_problems = len(hf["problems"])
            num_problems = min(episode, total_problems)

            # Initialize lists for data
            self.raw_data_capacity = []
            self.raw_data_demand = []
            self.raw_data_cost = []
            self.raw_data_node_flag = []
            self.raw_data_problems = []  # Store the problem indices

            # Load problems
            for i in tqdm(range(num_problems), desc=f"Loading {self.mode} data"):
                # Get problem data
                problem = hf["problems"][i]
                demand = hf["demands"][i]
                capacity = hf["capacities"][i]
                distance = hf["distances"][i]
                node_flag = hf["node_flags"][i]

                # Convert node_flag to two-column format
                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_problems.append(problem.tolist())
                self.raw_data_capacity.append(int(capacity))
                self.raw_data_demand.append(demand.tolist())
                self.raw_data_cost.append(float(distance))
                self.raw_data_node_flag.append(node_flag)

            # Convert to tensors
            self.raw_data_problems = torch.tensor(
                self.raw_data_problems, requires_grad=False
            ).to(self.device)
            self.raw_data_capacity = torch.tensor(
                self.raw_data_capacity, requires_grad=False
            ).to(self.device)
            self.raw_data_demand = torch.tensor(
                self.raw_data_demand, requires_grad=False
            ).to(self.device)
            self.raw_data_cost = torch.tensor(
                self.raw_data_cost, requires_grad=False
            ).to(self.device)
            self.raw_data_node_flag = torch.tensor(
                self.raw_data_node_flag, requires_grad=False
            ).to(self.device)

        logging.info(
            f"Loading {self.mode} dataset done! Loaded {len(self.raw_data_capacity)} problems."
        )

    def shuffle_data(self):
        # Shuffle the training set data
        index = torch.randperm(len(self.raw_data_problems)).long()
        self.raw_data_problems = self.raw_data_problems[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]

    def vrp_whole_and_solution_subrandom_inverse(self, solution):
        """Randomly flips the entire solution or individual subtours"""
        # Check if we have valid input
        if solution.shape[0] == 0 or solution.shape[1] == 0:
            return solution

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])
            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1. Find the number of subtours in each instance
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        # Count depot visits in each solution
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        # Skip processing if there are no depot visits
        if torch.min(visit_depot_num) == 0:
            return solution

        all_subtour_num = torch.sum(visit_depot_num)

        # Add a sentinel value to mark the end
        fake_solution = torch.cat(
            (solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1
        )
        start_from_depot = fake_solution.nonzero()

        # Handle empty solutions
        if len(start_from_depot) == 0:
            return solution

        start_from_depot_1 = start_from_depot[:, 1]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)
        sub_tours_length = start_from_depot_2 - start_from_depot_1

        # Handle case with no valid subtours
        if len(sub_tours_length) == 0:
            return solution

        max_subtour_length = torch.max(sub_tours_length)

        # 2. Extract and process subtours
        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        # Handle case with no depot visits
        if len(start_from_depot2) == 0 or len(start_from_depot3) == 0:
            return solution

        # Repeat solutions for each depot visit
        repeat_solutions_node = solution[:, :, 0].repeat_interleave(
            visit_depot_num, dim=0
        )
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        # Create masks for subtour extraction
        x1 = (
            torch.arange(double_repeat_solution_node.shape[1], device=self.device)[
                None, :
            ].repeat(len(repeat_solutions_node), 1)
            >= start_from_depot2[:, 1][:, None]
        )
        x2 = (
            torch.arange(double_repeat_solution_node.shape[1], device=self.device)[
                None, :
            ].repeat(len(repeat_solutions_node), 1)
            <= start_from_depot3[:, 1][:, None]
        )
        x3 = (x1 * x2).long()
        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1], device=self.device)[
            None, :
        ].repeat(len(repeat_solutions_node), 1) < (
            start_from_depot2[:, 1][:, None] + max_subtour_length
        )
        x5 = x1 * x4

        # Skip if no valid subtours
        if not x5.any():
            return solution

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 3. Randomly flip subtours
        clockwise_or_not = torch.rand(len(sub_tours_padding), device=self.device)
        clockwise_or_not_bool = clockwise_or_not.le(0.5)

        # Only flip if we have valid subtours to flip
        if clockwise_or_not_bool.any():
            sub_tours_padding[clockwise_or_not_bool] = torch.flip(
                sub_tours_padding[clockwise_or_not_bool], dims=[1]
            )

        # 4. Map back to original solution
        sub_tourss_back = sub_tourss.clone()
        sub_tourss_back[x5] = sub_tours_padding.ravel()

        # Skip if no valid values
        if not sub_tourss_back.gt(0.1).any():
            return solution

        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(
            batch_size, problem_size
        )
        solution_flip = torch.cat(
            (solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2
        )

        return solution_flip

    def vrp_whole_and_solution_subrandom_shift_V2inverse(self, solution):
        """
        For each instance, shift randomly so that different end_with depot nodes can reach the last digit.
        """
        # Check if we have valid input
        if solution.shape[0] == 0 or solution.shape[1] == 0:
            return solution

        problem_size = solution.shape[1]
        batch_size = solution.shape[0]

        # Find depot visits
        start_from_depot = solution[:, :, 1].nonzero()

        # Skip if no depot visits
        if len(start_from_depot) == 0:
            return solution

        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1

        # Fix all negative indices, not just the first one
        negative_indices = end_with_depot[:, 1] < 0
        end_with_depot[negative_indices, 1] = solution.shape[1] - 1

        end_with_depot[:, 1] = torch.roll(end_with_depot[:, 1], dims=0, shifts=-1)
        visit_depot_num = solution[:, :, 1].sum(1)

        # Skip if no depot visits
        if torch.min(visit_depot_num) == 0:
            return solution

        min_length = torch.min(visit_depot_num)

        # Generate random index for first node
        first_node_index = torch.randint(
            low=0, high=max(1, min_length.item()), size=[1], device=self.device
        )[0]

        # Use PyTorch for the triangular matrix operation to avoid NumPy conversion
        batch_size_int = len(visit_depot_num)
        temp_tri = torch.triu(
            torch.ones((batch_size_int, batch_size_int), device=self.device), diagonal=1
        )

        # Perform matrix multiplication in PyTorch
        temp_index_torch = torch.matmul(visit_depot_num.float(), temp_tri).long()

        pick_end_with_depot_index = temp_index_torch + first_node_index

        # Ensure indices are within bounds
        valid_indices = pick_end_with_depot_index < len(end_with_depot)
        if not valid_indices.all():
            pick_end_with_depot_index = torch.clamp(
                pick_end_with_depot_index, 0, len(end_with_depot) - 1
            )

        pick_end_with_depot_ = end_with_depot[pick_end_with_depot_index][:, 1]
        first_index = pick_end_with_depot_
        end_indeex = pick_end_with_depot_ + problem_size

        # Create indices for the double solution
        index = torch.arange(2 * problem_size, device=self.device)[None, :].repeat(
            batch_size, 1
        )
        x1 = index > first_index[:, None]
        x2 = index <= end_indeex[:, None]
        x3 = x1.int() * x2.int()

        # Skip if no valid indices
        if not x3.gt(0.5).any():
            return solution

        double_solution = solution.repeat(1, 2, 1)

        # Use masked_select for more efficient selection
        mask = x3.gt(0.5)[:, :, None].repeat(1, 1, 2)
        selected = double_solution[mask].view(batch_size, problem_size, 2)

        return selected

    def sampling_subpaths(self, problem, solution, fixed_length=None):
        """
        Sample subpaths from the solution for training.
        Handles edge cases and improves robustness.
        """
        # Check if we have valid input
        if problem.shape[0] <= 1 or solution.shape[0] == 0:
            return problem, solution

        # 1. Extract subtour
        problems_size = problem.shape[0] - 1  # Excluding depot
        embedding_size = problem.shape[1]

        # Use fixed length if provided, otherwise random
        if fixed_length is not None:
            length_of_subpath = min(
                fixed_length, problems_size
            )  # Ensure length doesn't exceed problem size
        else:
            # Random length of subpath between 4 and problem size
            length_of_subpath = torch.randint(
                low=min(4, problems_size),
                high=problems_size + 1,
                size=[1],
                device=self.device,
            )[0]

        # Apply data augmentation with error handling
        try:
            solution = self.vrp_whole_and_solution_subrandom_inverse(
                solution.unsqueeze(0)
            ).squeeze(0)

            solution = self.vrp_whole_and_solution_subrandom_shift_V2inverse(
                solution.unsqueeze(0)
            ).squeeze(0)
        except Exception as e:
            logging.warning(
                f"Error in data augmentation: {e}. Using original solution."
            )

        # Find points that start from depot
        depot_visits = solution[:, 1].nonzero()

        # Handle case with no depot visits
        if len(depot_visits) == 0:
            # If no depot visits, use the entire solution as is
            return problem, solution

        start_from_depot = depot_visits.squeeze(1)
        end_with_depot = start_from_depot.clone() - 1

        # Fix all negative indices
        negative_indices = end_with_depot < 0
        end_with_depot[negative_indices] = solution.shape[0] - 1

        # Count depot visits
        visit_depot_num = torch.sum(solution[:, 1])

        # Handle case with no depot visits
        if visit_depot_num == 0:
            return problem, solution

        # Randomly select end point
        p = torch.rand(1, device=self.device)
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(
            select_end_with_depot_node_index
        ).long()

        # Ensure index is in bounds
        if select_end_with_depot_node_index >= len(end_with_depot):
            select_end_with_depot_node_index = len(end_with_depot) - 1

        # This is the point at which the instance is randomly selected with an end with depot
        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index]

        # Create double solution for circular handling
        double_solution = torch.cat((solution, solution), dim=0)
        select_end_with_depot_node = select_end_with_depot_node + problems_size

        # Create indices for subpath extraction
        offset = select_end_with_depot_node - length_of_subpath + 1
        indexx = torch.arange(length_of_subpath, device=self.device) + offset

        # Ensure indices are in bounds
        if torch.max(indexx) >= double_solution.shape[0]:
            indexx = torch.clamp(indexx, 0, double_solution.shape[0] - 1)

        sub_tour = double_solution[indexx, :]

        # Calculate the capacity of the first point
        start_index = indexx[0]

        # Process capacity constraints
        x1 = torch.arange(solution.shape[0], device=self.device) <= start_index
        before_is_via_depot_all = solution[:, 1] * x1
        visit_depot_num_2 = torch.sum(before_is_via_depot_all)

        # Update node indices for the subpath
        sub_solution_node = sub_tour[:, 0]

        # Handle empty subpaths
        if len(sub_solution_node) == 0:
            return problem, solution

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

        index_3 = torch.arange(embedding_size, dtype=torch.long, device=self.device)
        index_3 = index_3.repeat(length_of_subpath)

        new_data = problem[index_2, index_3].view(length_of_subpath, embedding_size)
        new_data = torch.cat((problem[0, :].unsqueeze(dim=0), new_data), dim=0)

        return new_data, sub_tour


def collate_batch(batch):
    # Collate function to handle batching of variable-sized problems
    problems = torch.stack([item["problem"] for item in batch])
    solutions = torch.stack([item["solution"] for item in batch])
    capacities = torch.stack([item["capacity"] for item in batch])

    return {"problems": problems, "solutions": solutions, "capacities": capacities}


def get_dataset(config, device):
    # Create dataset
    dataset = LEHDDataset(
        data_path=config["env"]["data_path"],
        episodes=config["episodes"],
        mode=config["mode"],
        sub_path=config["env"]["sub_path"],
        device=device,
    )
    return dataset, collate_batch
