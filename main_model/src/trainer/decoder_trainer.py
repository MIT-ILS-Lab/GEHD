import torch
import wandb
from tqdm import tqdm
from logging import getLogger

import torch.distributed as dist
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.data import DataLoader

from main_model.src.architecture.decoder_architecture import LEHD
from main_model.src.data.decoder_dataloader import (
    LEHDDataset,
    LEHDBatchSampler,
    vrp_collate_fn,
)
from main_model.src.utils.general_utils import *


class LEHDTrainer:
    def __init__(self, config):
        # save arguments
        self.global_step = 0

        self.model_params = config["model_params"]
        self.optimizer_params = config["optimizer_params"]
        self.trainer_params = config["trainer_params"]
        self.testing_params = config["testing_params"]
        self.wandb_params = config["wandb_params"]

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = config["logger_params"]["result_folder"]
        self.result_log = LogData()

        random_seed = 22
        torch.manual_seed(random_seed)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if self.rank == 0:
            wandb.init(
                project=self.wandb_params["project_name"],
                name=self.wandb_params["run_name"],
                dir=self.result_folder,
                config=config,
                mode="offline",
            )

        # Main Components
        self.model = LEHD(**self.model_params)

        # Create train dataset and dataloader
        self.dataset_train = LEHDDataset(
            data_path=self.trainer_params["env"]["data_path"],
            episodes=self.trainer_params["episodes"],
            mode="train",
            sub_path=self.trainer_params["env"]["sub_path"],
            device=self.device,
        )

        self.dataset_test = LEHDDataset(
            data_path=self.testing_params["env"]["data_path"],
            episodes=self.testing_params["episodes"],
            mode="test",
            sub_path=self.testing_params["env"]["sub_path"],
            device=self.device,
        )

        # Use custom batch sampler for the train dataloader
        dataset_size_train = len(self.dataset_train)
        problem_size_train = self.dataset_train.raw_data_nodes.shape[1] - 1
        batch_sampler_train = LEHDBatchSampler(
            dataset_size_train,
            problem_size_train,
            self.trainer_params["batch_size"],
        )
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=vrp_collate_fn,
            num_workers=self.trainer_params["num_workers"],
        )

        # Use custom batch sampler for the test dataloader
        dataset_size_test = len(self.dataset_test)
        problem_size_test = self.dataset_test.raw_data_nodes.shape[1] - 1
        batch_sampler_test = LEHDBatchSampler(
            dataset_size_test,
            problem_size_test,
            self.testing_params["batch_size"],
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_sampler=batch_sampler_test,
            collate_fn=vrp_collate_fn,
            num_workers=self.testing_params["num_workers"],
        )

        # Initialize optimizer and scheduler
        self.optimizer = Optimizer(
            self.model.parameters(), **self.optimizer_params["optimizer"]
        )
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])

        # Restore
        self.start_epoch = 1
        model_load = self.trainer_params["model_load"]
        if model_load["enable"]:
            self.logger.info("Loading saved model...")
            checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.result_log.set_raw_data(checkpoint["result_log"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = model_load["epoch"] - 1
            self.logger.info("Saved model loaded!")

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)

        save_gap = []
        epochs_range = range(self.start_epoch, self.trainer_params["epochs"] + 1)
        with tqdm(epochs_range) as pbar:
            for epoch in pbar:
                # Train
                train_loss = self.train_epoch(epoch, pbar)
                self.result_log.append("train_loss", epoch, train_loss)

                # LR Decay
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

                wandb.log({"train/lr": current_lr, "epoch": epoch})

                # Logs & Checkpoint
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                    epoch, self.trainer_params["epochs"]
                )
                self.logger.info(
                    f"Epoch {epoch:3d}/{self.trainer_params['epochs']:3d}: Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}]"
                )

                all_done = epoch == self.trainer_params["epochs"]

                # Testing
                if all_done or (epoch % self.testing_params["test_interval"] == 0):
                    test_score, test_student_score, test_gap = self.test_epoch(epoch)

                    # Log test results
                    self.result_log.append("test_score", epoch, test_score)
                    self.result_log.append(
                        "test_student_score", epoch, test_student_score
                    )
                    self.result_log.append("test_gap", epoch, test_gap)

                    # Log to wandb
                    wandb.log(
                        {
                            "test/optimal_score": test_score,
                            "test/student_score": test_student_score,
                            "test/gap_percentage": test_gap,
                            "epoch": epoch,
                        }
                    )

                    # Save test results
                    save_gap.append([test_score, test_student_score, test_gap])
                    np.savetxt(
                        self.result_folder + "/gap.txt",
                        save_gap,
                        delimiter=",",
                        fmt="%s",
                    )

                # Save checkpoint
                if all_done or (epoch % self.trainer_params["save_interval"] == 0):
                    self._save_checkpoint(epoch)

                if all_done:
                    self.logger.info(" *** Training Done *** ")
                    self.logger.info("Now, printing log array...")
                    util_print_log_array(self.logger, self.result_log)
                    wandb.finish()

    def _save_checkpoint(self, epoch):
        checkpoint_path = f"{self.result_folder}/checkpoint-{epoch}.pt"

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
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Log to wandb
        wandb.save(checkpoint_path)

    def _get_travel_distance(self, problems_, solution_):
        """
        Calculate the travel distance for a given solution.

        Args:
            problems_: The problem definition containing node coordinates
            solution_: The solution containing node indices and flags

        Returns:
            travel_distances: The calculated travel distances
        """
        problems = problems_[:, :, [0, 1]].clone()
        order_node = solution_[:, :, 0].clone()
        order_flag = solution_[:, :, 1].clone()
        travel_distances = self.cal_length(problems, order_node, order_flag)

        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        """
        Calculate the length of routes based on the solution.

        Args:
            problems: Tensor containing node coordinates [batch_size, num_nodes, 2]
            order_node: Tensor containing node indices in the solution [batch_size, solution_length]
            order_flag: Tensor containing flags (0: customer, 1: depot) [batch_size, solution_length]

        Returns:
            total_distance: Total distance for each solution in the batch
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

    def train_epoch(self, epoch, pbar):
        loss_AM = AverageMeter()
        self.model.train()

        for batch_data in tqdm(self.dataloader_train):
            problems = batch_data["problems"].to(self.device)
            solutions = batch_data["solutions"].to(self.device)
            capacities = batch_data["capacities"].to(self.device).float()

            batch_size = problems.size(0)

            # Process the batch
            avg_loss = self.train_step(problems, solutions, capacities)
            loss_AM.update(avg_loss, batch_size)

            wandb.log({"train/loss": avg_loss}, step=self.global_step)

            pbar.set_postfix({"train_loss": f"{loss_AM.avg:.4f}"})

            self.global_step += 1

        # Log final epoch stats
        self.logger.info(f"Epoch {epoch:3d}: Train (100.0%)  Loss: {loss_AM.avg:.4f}")

        return loss_AM.avg

    def train_step(self, problems, solutions, capacities):
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

                # Backpropagation
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

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

        # Calculate final scores (would implement cal_length method here)
        # For now returning placeholder values
        return loss_mean.item()

    def test_epoch(self, epoch, pbar):
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        gap_AM = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            for batch_data in self.dataloader_test:
                problems = batch_data["problems"].to(self.device)
                solutions = batch_data["solutions"].to(self.device)
                capacities = batch_data["capacities"].to(self.device).float()

                batch_size = problems.size(0)

                # Process the batch
                avg_score, score_student_mean = self.test_step(
                    problems, solutions, capacities
                )

                # Calculate gap as percentage
                gap = (score_student_mean - avg_score) / avg_score * 100

                score_AM.update(avg_score, batch_size)
                score_student_AM.update(score_student_mean, batch_size)
                gap_AM.update(gap, batch_size)

                pbar.set_postfix({"test_gap": f"{gap_AM.avg:.2f}%"})

            # Log final test stats
            self.logger.info(
                f"Epoch {epoch:3d}: Test (100.0%)  Optimal: {score_AM.avg:.4f}, Student: {score_student_AM.avg:.4f}, Gap: {gap_AM.avg:.4f}%"
            )

        self.model.train()
        return score_AM.avg, score_student_AM.avg, gap_AM.avg

    def test_step(self, problems, solutions, capacities):
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

        # Optional: Implement RRC (Randomized Reconstruction) if needed
        # This would be similar to the budget loop in the original code

        self.logger.info(
            "Test Results - Optimal Score: {:.4f}, Student Score: {:.4f}, Gap: {:.4f}%".format(
                optimal_length.mean().item(),
                current_best_length.mean().item(),
                (
                    (current_best_length.mean() - optimal_length.mean())
                    / optimal_length.mean()
                ).item()
                * 100,
            )
        )

        return (
            optimal_length.mean().item(),
            current_best_length.mean().item(),
        )
