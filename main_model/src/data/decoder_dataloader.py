import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


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
        # Return the length of the underlying batch sampl
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

    def __len__(self):
        return self.episodes

    def __getitem__(self, args):
        # Get a single problem instance
        idx = args[0]
        fixed_length = args[1]

        nodes = self.raw_data_nodes[idx]
        capacity = self.raw_data_capacity[idx]
        demand = self.raw_data_demand[idx]
        solution = self.raw_data_node_flag[idx]

        # Create the problem representation
        capacity_expanded = capacity.unsqueeze(0).repeat(solution.shape[0] + 1)
        problem = torch.cat(
            (nodes, demand.unsqueeze(-1), capacity_expanded.unsqueeze(-1)), dim=1
        )

        # Apply subpath sampling if needed, using the provided fixed_length
        if self.sub_path:
            problem, solution = self.sampling_subpaths(
                problem, solution, fixed_length=fixed_length
            )

        return {"problem": problem, "solution": solution, "capacity": capacity}

    def load_raw_data(self, episode=1000000):
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        if self.mode == "train":
            self.raw_data_nodes_1 = []
            self.raw_data_capacity_1 = []
            self.raw_data_demand_1 = []
            self.raw_data_cost_1 = []
            self.raw_data_node_flag_1 = []

            for line in tqdm(
                open(self.data_path, "r").readlines()[0 : int(0.5 * episode)],
                ascii=True,
            ):
                line = line.split(",")

                depot_index = int(line.index("depot"))
                customer_index = int(line.index("customer"))
                capacity_index = int(line.index("capacity"))
                demand_index = int(line.index("demand"))
                cost_index = int(line.index("cost"))
                node_flag_index = int(line.index("node_flag"))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [
                    [float(line[idx]), float(line[idx + 1])]
                    for idx in range(customer_index + 1, capacity_index, 2)
                ]

                loc = depot + customer
                capacity = int(float(line[capacity_index + 1]))

                if int(line[demand_index + 1]) == 0:
                    demand = [
                        int(line[idx]) for idx in range(demand_index + 1, cost_index)
                    ]
                else:
                    demand = [0] + [
                        int(line[idx]) for idx in range(demand_index + 1, cost_index)
                    ]

                cost = float(line[cost_index + 1])
                node_flag = [
                    int(line[idx]) for idx in range(node_flag_index + 1, len(line))
                ]
                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_nodes_1.append(loc)
                self.raw_data_capacity_1.append(capacity)
                self.raw_data_demand_1.append(demand)
                self.raw_data_cost_1.append(cost)
                self.raw_data_node_flag_1.append(node_flag)

            self.raw_data_nodes_1 = torch.tensor(
                self.raw_data_nodes_1, requires_grad=False
            ).to(self.device)
            self.raw_data_capacity_1 = torch.tensor(
                self.raw_data_capacity_1, requires_grad=False
            ).to(self.device)
            self.raw_data_demand_1 = torch.tensor(
                self.raw_data_demand_1, requires_grad=False
            ).to(self.device)
            self.raw_data_cost_1 = torch.tensor(
                self.raw_data_cost_1, requires_grad=False
            ).to(self.device)
            self.raw_data_node_flag_1 = torch.tensor(
                self.raw_data_node_flag_1, requires_grad=False
            ).to(self.device)

            # Load second half of data
            self.raw_data_nodes_2 = []
            self.raw_data_capacity_2 = []
            self.raw_data_demand_2 = []
            self.raw_data_cost_2 = []
            self.raw_data_node_flag_2 = []

            for line in tqdm(
                open(self.data_path, "r").readlines()[
                    int(0.5 * episode) : int(episode)
                ],
                ascii=True,
            ):
                line = line.split(",")

                depot_index = int(line.index("depot"))
                customer_index = int(line.index("customer"))
                capacity_index = int(line.index("capacity"))
                demand_index = int(line.index("demand"))
                cost_index = int(line.index("cost"))
                node_flag_index = int(line.index("node_flag"))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [
                    [float(line[idx]), float(line[idx + 1])]
                    for idx in range(customer_index + 1, capacity_index, 2)
                ]

                loc = depot + customer
                capacity = int(float(line[capacity_index + 1]))

                if int(line[demand_index + 1]) == 0:
                    demand = [
                        int(line[idx]) for idx in range(demand_index + 1, cost_index)
                    ]
                else:
                    demand = [0] + [
                        int(line[idx]) for idx in range(demand_index + 1, cost_index)
                    ]

                cost = float(line[cost_index + 1])
                node_flag = [
                    int(line[idx]) for idx in range(node_flag_index + 1, len(line))
                ]
                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_nodes_2.append(loc)
                self.raw_data_capacity_2.append(capacity)
                self.raw_data_demand_2.append(demand)
                self.raw_data_cost_2.append(cost)
                self.raw_data_node_flag_2.append(node_flag)

            self.raw_data_nodes_2 = torch.tensor(
                self.raw_data_nodes_2, requires_grad=False
            ).to(self.device)
            self.raw_data_capacity_2 = torch.tensor(
                self.raw_data_capacity_2, requires_grad=False
            ).to(self.device)
            self.raw_data_demand_2 = torch.tensor(
                self.raw_data_demand_2, requires_grad=False
            ).to(self.device)
            self.raw_data_cost_2 = torch.tensor(
                self.raw_data_cost_2, requires_grad=False
            ).to(self.device)
            self.raw_data_node_flag_2 = torch.tensor(
                self.raw_data_node_flag_2, requires_grad=False
            ).to(self.device)

            # Combine both halves
            self.raw_data_nodes = torch.cat(
                (self.raw_data_nodes_1, self.raw_data_nodes_2), dim=0
            )
            self.raw_data_capacity = torch.cat(
                (self.raw_data_capacity_1, self.raw_data_capacity_2), dim=0
            )
            self.raw_data_demand = torch.cat(
                (self.raw_data_demand_1, self.raw_data_demand_2), dim=0
            )
            self.raw_data_cost = torch.cat(
                (self.raw_data_cost_1, self.raw_data_cost_2), dim=0
            )
            self.raw_data_node_flag = torch.cat(
                (self.raw_data_node_flag_1, self.raw_data_node_flag_2), dim=0
            )

        elif self.mode == "test":
            self.raw_data_nodes = []
            self.raw_data_capacity = []
            self.raw_data_demand = []
            self.raw_data_cost = []
            self.raw_data_node_flag = []

            for line in tqdm(
                open(self.data_path, "r").readlines()[0:episode], ascii=True
            ):
                line = line.split(",")

                depot_index = int(line.index("depot"))
                customer_index = int(line.index("customer"))
                capacity_index = int(line.index("capacity"))
                demand_index = int(line.index("demand"))
                cost_index = int(line.index("cost"))
                node_flag_index = int(line.index("node_flag"))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [
                    [float(line[idx]), float(line[idx + 1])]
                    for idx in range(customer_index + 1, capacity_index, 2)
                ]

                loc = depot + customer
                capacity = int(float(line[capacity_index + 1]))

                if int(line[demand_index + 1]) == 0:
                    demand = [
                        int(line[idx]) for idx in range(demand_index + 1, cost_index)
                    ]
                else:
                    demand = [0] + [
                        int(line[idx]) for idx in range(demand_index + 1, cost_index)
                    ]

                cost = float(line[cost_index + 1])
                node_flag = [
                    int(line[idx]) for idx in range(node_flag_index + 1, len(line))
                ]
                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_nodes.append(loc)
                self.raw_data_capacity.append(capacity)
                self.raw_data_demand.append(demand)
                self.raw_data_cost.append(cost)
                self.raw_data_node_flag.append(node_flag)

            self.raw_data_nodes = torch.tensor(
                self.raw_data_nodes, requires_grad=False
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

        print(f"load raw dataset done!")

    def shuffle_data(self):
        # Shuffle the training set data
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]

    def vrp_whole_and_solution_subrandom_inverse(self, solution):
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

    def vrp_whole_and_solution_subrandom_shift_V2inverse(self, solution):
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
        temp_index_torch = torch.from_numpy(temp_index).long().to(self.device)

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

    def sampling_subpaths(self, problem, solution, fixed_length=None):
        # problem shape (V+1,4)
        # solution shape (V,2)

        # 1. Extract subtour
        problems_size = problem.shape[0] - 1  # Excluding depot
        embedding_size = problem.shape[1]

        # Use fixed length if provided, otherwise random
        if fixed_length is not None:
            length_of_subpath = fixed_length
        else:
            # Random length of subpath between 4 and problem size
            length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[
                0
            ]

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


def vrp_collate_fn(batch):
    # Collate function to handle batching of variable-sized problems
    problems = torch.stack([item["problem"] for item in batch])
    solutions = torch.stack([item["solution"] for item in batch])
    capacities = torch.stack([item["capacity"] for item in batch])

    return {"problems": problems, "solutions": solutions, "capacities": capacities}
