import time
import wandb
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main_model.src.utils.tracker import AverageTracker

from main_model.src.architecture.decoder_architecture import LEHD
from main_model.src.trainer.base_trainer import Solver
from main_model.src.data.decoder_dataloader import (
    LEHDBatchSampler,
    InfiniteLEHDBatchSampler,
    get_dataset,
)

logger = logging.getLogger(__name__)


class LEHDTrainer(Solver):
    def __init__(self, config, is_master=True):
        super().__init__(config, is_master)

        # Store trainer and testing params
        self.trainer_params = config["data"]["train"]
        self.testing_params = config["data"]["test"]

        # Set random seed
        torch.manual_seed(22)

    def get_model(self, config):
        # Create and return the LEHD model
        return LEHD(**config).to(self.device)

    def get_dataset(self, config):
        return get_dataset(config)

    def get_dataloader(self, config):
        # Override to use custom batch sampler
        dataset, collate_fn = self.get_dataset(config)

        # Determine if this is train or test config
        is_train = config["mode"] == "train"
        params = self.trainer_params if is_train else self.testing_params

        # Create batch sampler
        dataset_size = len(dataset)
        problem_size = dataset.raw_data_problems.shape[1] - 1
        batch_sampler = LEHDBatchSampler(
            dataset_size,
            problem_size,
            params["batch_size"],
            params["shuffle"],
        )

        # Wrap it with the infinite sampler
        infinite_batch_sampler = InfiniteLEHDBatchSampler(batch_sampler)

        # Create and return dataloader
        data_loader = DataLoader(
            dataset,
            batch_sampler=infinite_batch_sampler,
            collate_fn=collate_fn,
            num_workers=params["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
        return data_loader

    def train_epoch(self, epoch):
        self.model.train()

        tick = time.time()
        elapsed_time = dict()

        train_tracker_epoch = AverageTracker()

        for episode in range(1, len(self.train_loader) + 1):
            train_tracker = AverageTracker()

            # load data
            batch = self.train_iter.__next__()
            batch["iter_num"] = episode
            batch["epoch"] = epoch
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            elapsed_time["time/data"] = torch.Tensor([time.time() - tick])

            # forward and backward
            output = self.train_step(batch)

            # track the averaged tensors
            elapsed_time["time/batch"] = torch.Tensor([time.time() - tick])
            tick = time.time()
            output.update(elapsed_time)
            train_tracker.update(output)
            train_tracker_epoch.update(output)

            if (
                episode % 50 == 0
                and self.config["solver"]["empty_cache"]
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

            if self.log_per_iter > 0 and self.global_step % self.log_per_iter == 0:
                logger.info(
                    "Epoch {:3d}: Train {:3d}/{:3d} ({:5.1f}%) Loss: {:.4f} Time: {:.2f}".format(
                        epoch,
                        episode,
                        len(self.train_loader),
                        episode / len(self.train_loader) * 100,
                        output["train/loss"],
                        output["time/batch"].item() / 60,
                    )
                )

            self.global_step += 1

            if self.rank == 0:
                log_data = train_tracker.average()
                log_data["lr"] = self.optimizer.param_groups[0]["lr"]
                wandb.log(log_data, step=self.global_step)

        logger.info(" ")
        logger.info("*** Summary ***")
        logger.info(
            "Avg. Loss: {:.2f} Avg. Time: {:.2f} min".format(
                train_tracker_epoch.average()["train/loss"],
                train_tracker_epoch.average()["time/batch"] / 60,
            )
        )

    def test_epoch(self, epoch):
        self.model.eval()
        test_tracker = AverageTracker()

        tick = time.time()  # Start time for batch timing
        elapsed_time = dict()

        for episode in range(1, len(self.test_loader) + 1):  # Simple loop without tqdm
            # Load data
            batch = self.test_iter.__next__()
            batch["iter_num"] = episode
            batch["epoch"] = epoch

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            elapsed_time["time/data"] = torch.Tensor([time.time() - tick])

            # Forward pass using test_step
            with torch.no_grad():
                output = self.test_step(batch)

            elapsed_time["time/batch"] = torch.Tensor([time.time() - tick])
            tick = time.time()

            # Update tracker with metrics and timing
            output.update(elapsed_time)
            test_tracker.update(output)

            # Log per iteration if required
            if self.log_per_iter > 0 and self.global_step % self.log_per_iter == 0:
                logger.info(
                    "Epoch {:3d}: Test {:3d}/{:3d} ({:5.1f}%) Gap: {:.4f} Time: {:.2f}".format(
                        epoch,
                        episode,
                        len(self.test_loader),
                        (episode) / len(self.test_loader) * 100,
                        output["test/gap_percentage"],
                        output["time/batch"].item() / 60,
                    )
                )

        # Log final averaged metrics to wandb and console
        # Logg averages
        logger.info(" ")
        logger.info("*** Summary ***")
        logger.info(
            "Avg. Opt. Score: {:.2f}, Avg. St. Score: {:.2f}".format(
                test_tracker.average()["test/optimal_score"],
                test_tracker.average()["test/student_score"],
            )
        )
        logger.info(
            "Avg. Gap: {:.2f}%, Avg. Time {:.2f} min".format(
                test_tracker.average()["test/gap_percentage"],
                test_tracker.average()["time/batch"] / 60,
            )
        )
        logger.info(" ")

        if self.rank == 0:
            log_data = test_tracker.average()
            log_data["epoch"] = epoch
            wandb.log(log_data, step=self.global_step)

    def train_step(self, batch):
        # Extract data from batch
        solutions = batch["solutions"]
        capacities = batch["capacities"].float()

        loss_list = []
        # Training loop for constructing solution
        while (
            solutions.size(1) > 3
        ):  # if solutions.size(1) == 3, only start, destination and depot left
            logits_node, logits_flag = self.model(
                solutions,
                capacities,
                mode="train",
            )

            node_teacher = solutions[:, 1, 1].to(torch.int64)
            flag_teacher = solutions[:, 1, -1].to(torch.int64)

            # calculate the cross entropy loss for the node selection
            # TODO: Add int conversion somewhere deeper in the model/ loader
            loss_node = F.cross_entropy(
                logits_node, node_teacher
            )  # TODO: Pretty certain but check if 1) need to add -1 and 2) if softmax before
            loss_flag = F.cross_entropy(logits_flag, flag_teacher)

            loss = 0.5 * loss_node + 0.5 * loss_flag
            loss = loss

            # Backpropagate and update model
            self.model.zero_grad()
            loss.backward()

            self.clip_grad_norm()
            self.optimizer.step()

            # Update capacity in problems tensor directly
            # 1. If flag = 1, the vehicle returns to depot and capacity is refilled
            is_depot = flag_teacher == 1
            solutions[is_depot, :, 3] = capacities[is_depot, None]

            # 2. Get demands of selected nodes using gather
            selected_demands = solutions[:, 1, 2]

            # 3. If capacity is less than demand, capacity is refilled and flag is changed to 1
            # TODO: Does this ever happen since we are working with the teacher's solution?
            smaller_ = solutions[:, 0, 3] < selected_demands
            solutions[smaller_, :, 3] = capacities[smaller_, None]

            # 4. Subtract demand from capacity
            solutions[:, :, 3] = solutions[:, :, 3] - selected_demands[:, None]

            # 5. Update problems tensor for next step
            solutions = solutions[:, 1:, :]

            loss_list.append(loss)

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
                    capacities,
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
            "test/optimal_score": optimal_length.mean(),
            "test/student_score": current_best_length.mean(),
            "test/gap_percentage": gap,
        }

    def _get_travel_distance(self, problems_, solution_):
        """
        Calculate the travel distance for a given solution.
        """
        problems = problems_[:, :, 0].clone()
        order_node = solution_[:, :, 0].clone()
        order_flag = solution_[:, :, 1].clone()
        travel_distances = self.cal_length(problems, order_node, order_flag)

        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()

        problems = problems.int()

        # Get the dataset to access the geodesic matrix
        dataset = self.test_loader.dataset
        geodesic_matrix = dataset.geodesic_matrix.to(self.device)

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        order_loc = problems.gather(dim=1, index=order_node_)
        roll_loc = problems.gather(dim=1, index=roll_node)
        flag_loc = problems.gather(dim=1, index=order_flag_)

        order_lengths = geodesic_matrix[order_loc, flag_loc]

        order_flag_[:, 0] = 0

        flag_loc = problems.gather(dim=1, index=order_flag_)

        roll_lengths = geodesic_matrix[roll_loc, flag_loc]

        length = order_lengths.sum() + roll_lengths.sum()

        return length
