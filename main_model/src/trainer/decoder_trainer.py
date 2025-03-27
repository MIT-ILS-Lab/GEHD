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
    batch_size, seq_len = logits_node.shape
    batch_size_solutions = solutions.shape[0]

    if batch_size != batch_size_solutions:
        raise ValueError("Batch sizes of logits_node and solutions must match.")

    masked_logits_node = logits_node.clone()

    # Extract the indices from the 3D solutions tensor
    indices = solutions[:, 1:-2, 0].to(
        torch.int64
    )  # (batch_size, num_solutions, solution_length - 2)

    # Flatten the indices and batch indices for advanced indexing
    flat_indices = indices.view(
        batch_size, -1
    )  # (batch_size, num_solutions * (solution_length - 2))

    # Create a batch index tensor for advanced indexing
    batch_indices = (
        torch.arange(batch_size, device=flat_indices.device)
        .view(batch_size, 1)
        .expand(batch_size, flat_indices.shape[1])
    )

    # Create a mask with True everywhere
    mask = torch.ones(
        (batch_size, seq_len), dtype=torch.bool, device=logits_node.device
    )

    # Use advanced indexing to set the masked positions to False
    mask[batch_indices, flat_indices] = False

    # Apply the mask to logits_node
    masked_logits_node[mask] = torch.tensor(-float("inf"), device=logits_node.device)

    return masked_logits_node


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

            node_teacher = solutions[:, 1, 0].to(torch.int64)
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

            # mask all nodes except the unvisisetd nodes (solutions 1 to -2)
            masked_logits_node = mask_logits(logits_node, solutions)

            node_student = masked_logits_node.argmax(dim=1)
            flag_student = logits_flag.argmax(dim=1)

            # Update capacity in problems tensor directly
            # 1. If flag = 1, the vehicle returns to depot and capacity is refilled
            is_depot = flag_student == 1
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
            # TODO: Continue here, find the indixes where solutions[:,i, 0] == node_student
            mask = solutions[:, :, 0] == node_student.unsqueeze(1)

            start_node = solutions[mask]
            solutions = solutions[~mask]

            # rebatch the start_node and solutions
            start_node = start_node.view(batch_size, -1, start_node.size(1))
            solutions = solutions.view(batch_size, -1, solutions.size(1))

            # switch the solution and move the student node indexes to the first entry
            solutions = torch.cat((start_node, solutions), dim=1)

            selected_student_list = torch.cat(
                (selected_student_list, node_student.unsqueeze(1)), dim=1
            )
            selected_student_flag = torch.cat(
                (selected_student_flag, flag_student.unsqueeze(1)), dim=1
            )

        selected_student_list = torch.cat(
            (selected_student_list, solutions[:, -2, 0].unsqueeze(1)), dim=1
        )
        selected_student_flag = torch.cat(
            (selected_student_flag, solutions[:, -2, -1].unsqueeze(1)), dim=1
        )

        # Combine node and flag information for final solution
        best_select_node_list = torch.cat(
            (selected_student_list.unsqueeze(2), selected_student_flag.unsqueeze(2)),
            dim=2,
        )

        # Calculate optimal and student scores
        optimal_length = self._get_travel_distance(
            torch.cat(
                (
                    solutions_orig.unsqueeze(2),
                    solutions_orig.unsqueeze(2),
                ),
                dim=2,
            ),
        )
        current_best_length = self._get_travel_distance(best_select_node_list)

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

    def _get_travel_distance(self, solution_):
        """
        Calculate the travel distance for a given solution.
        """
        order_node = solution_[:, :, 0].clone()
        order_flag = solution_[:, :, 1].clone()
        travel_distances = self.cal_length(order_node, order_flag)

        return travel_distances

    def cal_length(self, order_node, order_flag):
        # problems:   [B,V+1,2]
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
        flag_loc = order_flag

        order_lengths = geodesic_matrix[order_loc, flag_loc]

        order_flag_[:, 0] = 0

        flag_loc = order_flag_

        roll_lengths = geodesic_matrix[roll_loc, flag_loc]

        length = order_lengths.sum() + roll_lengths.sum()

        return length
