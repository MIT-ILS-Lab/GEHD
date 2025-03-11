from logging import getLogger
import torch
from functools import partial
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.data import DataLoader

from main_model.src.architecture.decoder_architecture import LEHD
from main_model.src.data.decoder_dataloader import (
    LEHDDataset,
    FixedLengthBatchSampler,
    vrp_collate_fn,
    # fixed_length_vrp_collate_fn,
)
from main_model.src.utils.general_utils import *


class LEHDTrainer:
    def __init__(self, config):
        # save arguments
        self.env_params = config["env_params"]
        self.model_params = config["model_params"]
        self.optimizer_params = config["optimizer_params"]
        self.trainer_params = config["trainer_params"]

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = config["logger_params"]["result_folder"]
        self.result_log = LogData()
        random_seed = 22
        torch.manual_seed(random_seed)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main Components
        self.model = LEHD(**self.model_params)

        # Create train dataset and dataloader
        self.dataset_train = LEHDDataset(
            data_path=self.env_params["data_path"],
            episodes=self.trainer_params["train_episodes"],
            mode=self.env_params["mode"],
            sub_path=self.env_params["sub_path"],
            device=self.device,
        )

        # Use custom batch sampler instead of shuffle=True
        dataset_size = len(self.dataset_train)
        problem_size = self.dataset_train.raw_data_nodes.shape[1] - 1
        batch_sampler = FixedLengthBatchSampler(
            dataset_size, problem_size, self.trainer_params["train_batch_size"]
        )

        # Create a partial function for the collate_fn that includes the dataset
        # collate_fn = partial(fixed_length_vrp_collate_fn, dataset=self.dataset_train)

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_sampler=batch_sampler,
            collate_fn=vrp_collate_fn,
            num_workers=self.trainer_params["num_workers"],
        )

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
        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            self.logger.info(
                "================================================================="
            )

            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, train_score)
            self.result_log.append("train_student_score", epoch, train_student_score)
            self.result_log.append("train_loss", epoch, train_loss)

            # LR Decay
            self.scheduler.step()

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params["epochs"]
            )
            self.logger.info(
                "Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch,
                    self.trainer_params["epochs"],
                    elapsed_time_str,
                    remain_time_str,
                )
            )

            all_done = epoch == self.trainer_params["epochs"]
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]
            img_save_interval = self.trainer_params["logging"]["img_save_interval"]

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = "{}/latest".format(self.result_folder)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_1"],
                    self.result_log,
                    labels=["train_score"],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_2"],
                    self.result_log,
                    labels=["train_loss"],
                )

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = "{}/img/checkpoint-{}".format(self.result_folder, epoch)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_1"],
                    self.result_log,
                    labels=["train_score"],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_2"],
                    self.result_log,
                    labels=["train_loss"],
                )

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        self.model.train()

        total_batches = len(self.dataloader_train)
        processed_batches = 0

        for batch_data in self.dataloader_train:
            problems = batch_data["problems"].to(self.device)
            solutions = batch_data["solutions"].to(self.device)
            capacities = batch_data["capacities"].to(self.device).float()

            batch_size = problems.size(0)

            # Process the batch
            avg_score, score_student_mean, avg_loss = self._train_one_batch(
                problems, solutions, capacities, epoch
            )

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)

            processed_batches += 1

            self.logger.info(
                "Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_student: {:.4f},  Loss: {:.4f}".format(
                    epoch,
                    processed_batches,
                    total_batches,
                    100.0 * processed_batches / total_batches,
                    score_AM.avg,
                    score_student_AM.avg,
                    loss_AM.avg,
                )
            )

        # Log once, for each epoch
        self.logger.info(
            "Epoch {:3d}: Train (100.0%)  Score: {:.4f}, Score_student: {:.4f}, Loss: {:.4f}".format(
                epoch,
                score_AM.avg,
                score_student_AM.avg,
                loss_AM.avg,
            )
        )

        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    def _train_one_batch(self, problems, solutions, capacities, epoch):
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
        return 0, 0, loss_mean.item()
