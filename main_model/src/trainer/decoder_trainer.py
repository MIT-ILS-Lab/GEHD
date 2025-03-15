import torch
import wandb

from torch.utils.data import DataLoader

from main_model.src.architecture.decoder_architecture import LEHD
from main_model.src.data.decoder_dataloader import (
    LEHDBatchSampler,
    InfiniteLEHDBatchSampler,
    get_dataset,
)
from main_model.src.utils.general_utils import LogData, TimeEstimator
from main_model.src.trainer.base_trainer import Solver


class LEHDTrainer(Solver):
    def __init__(self, config, is_master=True):
        super().__init__(config, is_master)

        # Store trainer and testing params
        self.trainer_params = config["data"]["train"]
        self.testing_params = config["data"]["test"]

        # Additional tracking specific to LEHD
        self.result_log = LogData()
        self.time_estimator = TimeEstimator()

        # Set random seed
        torch.manual_seed(22)

    def get_model(self, config):
        # Create and return the LEHD model
        return LEHD(**config)

    def get_dataset(self, config):
        return get_dataset(config, self.device)

    def get_dataloader(self, config):
        # Override to use custom batch sampler
        dataset, collate_fn = self.get_dataset(config)

        # Determine if this is train or test config
        is_train = config["mode"] == "train"
        params = self.trainer_params if is_train else self.testing_params

        # Create batch sampler
        dataset_size = len(dataset)
        problem_size = dataset.raw_data_nodes.shape[1] - 1
        batch_sampler = LEHDBatchSampler(
            dataset_size,
            problem_size,
            params["batch_size"],
        )

        # Wrap it with the infinite sampler
        infinite_batch_sampler = InfiniteLEHDBatchSampler(batch_sampler)

        # Create and return dataloader
        data_loader = DataLoader(
            dataset,
            batch_sampler=infinite_batch_sampler,
            collate_fn=collate_fn,
            num_workers=params["num_workers"],
        )
        return data_loader

    def train_step(self, batch):
        # Extract data from batch
        problems = batch["problems"]
        solutions = batch["solutions"]
        capacities = batch["capacities"].float()

        # Initialize state tracking
        batch_size = problems.size(0)
        problem_size = problems.size(1) - 1  # Excluding depot

        # Initialize selected node lists and flags
        selected_node_list = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )
        selected_teacher_flag = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )
        selected_student_list = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )
        selected_student_flag = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )

        # Track current capacity
        current_capacity = capacities.clone()

        loss_list = []
        current_step = 0

        # Training loop for constructing solution
        while current_step < problem_size:
            if current_step == 0:
                # First step - use teacher's first selection
                selected_teacher = solutions[:, 0, 0]
                selected_flag_teacher = solutions[:, 0, 1]
                selected_student = selected_teacher.clone()
                selected_flag_student = selected_flag_teacher.clone()
                loss_mean = torch.tensor(0, device=self.device)
            else:
                # Use model to predict next node
                (
                    loss_node,
                    selected_teacher,
                    selected_student,
                    selected_flag_teacher,
                    selected_flag_student,
                ) = self.model(
                    problems,
                    selected_node_list,
                    solutions,
                    current_step,
                    raw_data_capacity=capacities,
                    mode="train",
                )

                loss_mean = loss_node

            # Update capacity based on selection
            # Handle depot returns (capacity refill)
            is_depot = selected_flag_teacher == 1
            current_capacity[is_depot] = capacities[is_depot]

            # Get demands of selected nodes
            selected_demands = torch.gather(
                problems[:, :, 2], 1, selected_teacher.unsqueeze(1)
            ).squeeze(1)

            # Check if capacity is less than demand, refill if needed
            smaller_ = current_capacity < selected_demands
            selected_flag_teacher[smaller_] = 1
            current_capacity[smaller_] = capacities[smaller_]

            # Subtract demand from capacity
            current_capacity = current_capacity - selected_demands

            # Update tracking lists
            selected_node_list = torch.cat(
                (selected_node_list, selected_teacher.unsqueeze(1)), dim=1
            )
            selected_teacher_flag = torch.cat(
                (selected_teacher_flag, selected_flag_teacher.unsqueeze(1)), dim=1
            )
            selected_student_list = torch.cat(
                (selected_student_list, selected_student.unsqueeze(1)), dim=1
            )
            selected_student_flag = torch.cat(
                (selected_student_flag, selected_flag_student.unsqueeze(1)), dim=1
            )

            current_step += 1
            loss_list.append(loss_mean)

        # Calculate final loss
        loss_mean = torch.stack(loss_list).mean()

        # Return output dictionary for tracker
        return {"train/loss": loss_mean}

    def test_step(self, batch):
        # Extract data from batch
        problems = batch["problems"]
        solutions = batch["solutions"]
        capacities = batch["capacities"].float()

        # Initialize state tracking
        batch_size = problems.size(0)
        problem_size = problems.size(1) - 1  # Excluding depot

        # Initialize selected node lists and flags
        selected_node_list = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )
        selected_teacher_flag = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )
        selected_student_list = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )
        selected_student_flag = torch.zeros(
            (batch_size, 0), dtype=torch.long, device=self.device
        )

        # Track current capacity
        current_capacity = capacities.clone()
        current_step = 0

        # Store original problem for calculating distances later
        origin_problem = problems.clone().detach()

        # First pass - get initial solution
        while current_step < problem_size:
            if current_step == 0:
                # First step - use teacher's first selection
                selected_teacher = solutions[:, 0, 0]
                selected_flag_teacher = solutions[:, 0, 1]
                selected_student = selected_teacher.clone()
                selected_flag_student = selected_flag_teacher.clone()
            else:
                # Use model to predict next node
                (
                    _,
                    selected_teacher,
                    selected_student,
                    selected_flag_teacher,
                    selected_flag_student,
                ) = self.model(
                    problems,
                    selected_node_list,
                    solutions,
                    current_step,
                    raw_data_capacity=capacities,
                    mode="test",
                )

            # Update capacity based on selection
            # Handle depot returns (capacity refill)
            is_depot = selected_flag_student == 1
            current_capacity[is_depot] = capacities[is_depot]

            # Get demands of selected nodes
            selected_demands = torch.gather(
                problems[:, :, 2], 1, selected_student.unsqueeze(1)
            ).squeeze(1)

            # Check if capacity is less than demand, refill if needed
            smaller_ = current_capacity < selected_demands
            selected_flag_student[smaller_] = 1
            current_capacity[smaller_] = capacities[smaller_]

            # Subtract demand from capacity
            current_capacity = current_capacity - selected_demands

            # Update tracking lists
            selected_node_list = torch.cat(
                (selected_node_list, selected_student.unsqueeze(1)), dim=1
            )
            selected_teacher_flag = torch.cat(
                (selected_teacher_flag, selected_flag_teacher.unsqueeze(1)), dim=1
            )
            selected_student_list = torch.cat(
                (selected_student_list, selected_student.unsqueeze(1)), dim=1
            )
            selected_student_flag = torch.cat(
                (selected_student_flag, selected_flag_student.unsqueeze(1)), dim=1
            )

            current_step += 1

        # Combine node and flag information for final solution
        best_select_node_list = torch.cat(
            (selected_student_list.unsqueeze(2), selected_student_flag.unsqueeze(2)),
            dim=2,
        )

        # Calculate optimal and student scores
        optimal_length = self._get_travel_distance(
            origin_problem,
            torch.cat(
                (
                    solutions[:, :problem_size, 0].unsqueeze(2),
                    solutions[:, :problem_size, 1].unsqueeze(2),
                ),
                dim=2,
            ),
        )
        current_best_length = self._get_travel_distance(
            origin_problem, best_select_node_list
        )

        # Calculate gap as percentage
        gap = (
            (current_best_length.mean() - optimal_length.mean())
            / optimal_length.mean()
            * 100
        )

        # Return output dictionary for tracker
        return {
            "test/loss": current_best_length.mean(),
            "test/optimal_score": optimal_length.mean(),
            "test/student_score": current_best_length.mean(),
            "test/gap_percentage": gap,
        }

    def _get_travel_distance(self, problems_, solution_):
        """
        Calculate the travel distance for a given solution.
        """
        problems = problems_[:, :, [0, 1]].clone()
        order_node = solution_[:, :, 0].clone()
        order_flag = solution_[:, :, 1].clone()
        travel_distances = self.cal_length(problems, order_node, order_flag)

        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        """
        Calculate the length of routes based on the solution.
        """
        batch_size = problems.size(0)

        # Get coordinates of the depot (first node)
        depot = problems[:, 0, :].clone()

        # Initialize total distance
        total_distance = torch.zeros(batch_size, device=self.device)

        # Initialize current position as depot
        current_position = depot.clone()

        # Iterate through the solution
        for i in range(order_node.size(1)):
            # Get coordinates of the next node
            next_node_idx = order_node[:, i].unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)
            next_position = torch.gather(problems, 1, next_node_idx).squeeze(1)

            # Calculate distance to next node
            distance_to_next = torch.sqrt(
                torch.sum((current_position - next_position) ** 2, dim=1)
            )
            total_distance += distance_to_next

            # Update current position
            current_position = next_position.clone()

            # If returning to depot, update current position to depot
            is_depot = order_flag[:, i] == 1
            current_position[is_depot] = depot[is_depot]

        return total_distance

    def save_checkpoint(self, epoch):
        # Call parent method
        super().save_checkpoint(epoch)

        # Additional LEHD-specific checkpoint saving
        checkpoint_path = f"{self.logdir}/checkpoint-{epoch}.pt"

        # Create checkpoint dictionary
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "result_log": self.result_log.get_raw_data(),
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Log to wandb
        if self.rank == 0:
            wandb.save(checkpoint_path)
