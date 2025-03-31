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


def mask_logits(logits_node, solutions):
    batch_size, num_candidates = logits_node.shape

    # Create a mask with True everywhere
    mask = torch.ones(
        (batch_size, num_candidates), dtype=torch.bool, device=logits_node.device
    )

    # Get indices of nodes that have already been visited
    visited_nodes = solutions[:, 0, 0].unsqueeze(1)  # Current node

    # Check which candidates match visited nodes
    candidates = solutions[:, 1:-2, 0]
    for i in range(batch_size):
        for j in range(num_candidates):
            if candidates[i, j] in visited_nodes[i]:
                mask[i, j] = False

    # Apply the mask to logits_node
    masked_logits = logits_node.clone()
    masked_logits[~mask] = torch.tensor(-float("inf"), device=logits_node.device)

    return masked_logits


class LEHDTrainer(Solver):
    def __init__(self, config, is_master=True):
        super().__init__(config, is_master)

        # Store trainer and testing params
        self.trainer_params = config["data"]["train"]
        self.testing_params = config["data"]["test"]

        self.city_indices = None

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

        if self.city_indices is None:
            self.city_indices = dataset.city_indices
        else:
            assert torch.equal(
                self.city_indices, dataset.city_indices
            ), "City indices do not match"

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

    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_lr_scheduler()
        self.configure_log()
        self.load_checkpoint()

        self.time_estimator.reset(self.start_epoch)

        self.model.encoder.prepare_embedding(self.city_indices)

        for epoch in range(self.start_epoch, self.config["solver"]["max_epoch"] + 1):
            logger.info(
                f'====================  EPOCH {epoch:3d}/{self.config["solver"]["max_epoch"]:3d}  ===================='
            )
            # training epoch
            self.train_epoch(epoch)

            # update learning rate
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
            self.summary_writer.add_scalar("train/lr", lr[0], epoch)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.config["solver"]["max_epoch"]
            )
            logger.info(
                "Elapsed: {}, Remaining: {}".format(elapsed_time_str, remain_time_str)
            )
            logger.info(" ")

            # testing epoch
            if epoch % self.config["solver"]["test_every_epoch"] == 0:
                logger.info(
                    f'-------------------  TESTING {epoch:3d}/{self.config["solver"]["max_epoch"]:3d}  -------------------'
                )
                self.test_epoch(epoch)

            # checkpoint
            if epoch % self.config["solver"]["save_every_epoch"] == 0:
                self.save_checkpoint(epoch)
                logger.info("Saved checkpoint to %s" % self.ckpt_dir)

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
            )

            node_teacher = torch.zeros(solutions.size(0), dtype=torch.int64)
            flag_teacher = solutions[:, 1, -1].to(torch.int64)

            # Calculate loss with the position indices
            loss_node = F.cross_entropy(logits_node, node_teacher)
            loss_flag = F.cross_entropy(logits_flag, flag_teacher)

            loss = loss_node + loss_flag

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
        solutions = batch["solutions"]
        capacities = batch["capacities"].float()

        batch_size = solutions.size(0)

        solutions_orig = solutions.clone()

        start_node = solutions[:, 0, :]

        # initialize selected node student list with an index of solutions[:, -1, 0]
        selected_student_list = start_node[:, 0].unsqueeze(1)
        selected_student_flag = solutions[:, 0, -1].to(torch.int64).unsqueeze(1)
        # Training loop for constructing solution
        while (
            solutions.size(1) > 3
        ):  # if solutions.size(1) == 3, only start, destination and depot left
            logits_node, logits_flag = self.model(
                solutions,
                capacities,
            )

            # Select the highest scoring candidate
            node_indices = logits_node.argmax(dim=1)
            flag_student = logits_flag.argmax(dim=1)

            # Update capacity in problems tensor directly
            solutions = solutions[:, 1:, :]

            # 1. If flag = 1, the vehicle returns to depot and capacity is refilled
            is_depot = flag_student == 1
            solutions[is_depot, :, 3] = capacities[is_depot, None]

            # 2. Get demands of selected nodes using gather
            selected_demands = torch.gather(
                solutions[:, :, 2], dim=1, index=node_indices.unsqueeze(1)
            ).squeeze(1)

            # 3. If capacity is less than demand, capacity is refilled and flag is changed to 1
            smaller_ = solutions[:, 0, 3] < selected_demands
            solutions[smaller_, :, 3] = capacities[smaller_, None]
            flag_student[smaller_] = 1

            # 4. Subtract demand from capacity
            solutions[:, :, 3] = solutions[:, :, 3] - selected_demands[:, None]

            # 5. Update problems tensor for next step
            batch_size, num_rows, num_columns = solutions.shape

            # Create a range of row indices for each batch
            row_indices = (
                torch.arange(num_rows, device=solutions.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Create a mask for the rows that are not in node_indices
            mask = row_indices != node_indices.unsqueeze(1)

            # Gather the rows corresponding to node_indices
            source_rows = torch.gather(
                solutions,
                dim=1,
                index=node_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_columns),
            )

            # Gather the remaining rows
            remaining_rows = solutions[mask].view(batch_size, -1, num_columns)

            # Concatenate the rows, with the selected rows at the front
            solutions = torch.cat((source_rows, remaining_rows), dim=1)

            node_student = source_rows[:, :, 0]

            selected_student_list = torch.cat(
                (selected_student_list, node_student), dim=1
            )
            selected_student_flag = torch.cat(
                (selected_student_flag, flag_student.unsqueeze(1)), dim=1
            )

        # Add target location to the solution
        selected_student_list = torch.cat(
            (selected_student_list, solutions[:, -2, 0].unsqueeze(1)), dim=1
        )
        selected_student_flag = torch.cat(
            (selected_student_flag, solutions[:, -2, -1].unsqueeze(1)), dim=1
        )

        # Calculate optimal and student scores
        optimal_length = self.get_travel_distance(
            solutions_orig[:, :-1, 0]
            .clone()
            .to(torch.int64),  # exclude depot from the solution vector
            solutions_orig[:, :-1, 4]
            .clone()
            .to(torch.int64),  # exclude depot from the solution vector
        )
        current_best_length = self.get_travel_distance(
            selected_student_list.clone().to(torch.int64),
            selected_student_flag.clone().to(torch.int64),
        )

        # Calculate gap as percentage
        gap = 100 * ((current_best_length - optimal_length) / optimal_length).mean()

        # Return output dictionary for tracker
        return {
            "test/optimal_score": optimal_length.mean(),
            "test/student_score": current_best_length.mean(),
            "test/gap_percentage": gap,
        }

    def get_travel_distance(self, order_node, order_flag):
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()

        # Get the dataset to access the geodesic matrix
        dataset = self.test_loader.dataset
        geodesic_matrix = dataset.geodesic_matrix.to(self.device)

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        order_loc = order_node_
        roll_loc = roll_node
        flag_loc = order_flag_

        order_lengths = geodesic_matrix[order_loc, flag_loc]

        order_flag_[:, 0] = 0

        flag_loc = order_flag_

        roll_lengths = geodesic_matrix[roll_loc, flag_loc]

        length = order_lengths.sum(dim=1) + roll_lengths.sum(dim=1)

        return length
