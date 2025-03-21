import os
import time
import random
import numpy as np
from tqdm import tqdm
import wandb
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from main_model.src.utils.sampler import InfSampler
from main_model.src.utils.tracker import AverageTracker
from main_model.src.utils.lr_scheduler import get_lr_scheduler

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, config, is_master=True):
        self.config = config
        self.is_master = is_master
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.disable_tqdm = not (is_master and self.config["solver"]["progress_bar"])
        self.start_epoch = 1

        self.slurm_job_id = os.getenv("SLURM_JOB_ID")
        self.slurm_job_name = os.getenv("SLURM_JOB_NAME")
        logger.info(f"slurm configs: {self.slurm_job_id}, {self.slurm_job_name}")

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.clip_grad = self.config["solver"]["clip_grad"]

        self.batch_backprop = (
            True  # Set to False to avoid backpropagation after each step
        )

        if self.slurm_job_name == "JupyterNotebook":
            self.log_wandb = False
        else:
            self.log_wandb = True
        logger.info(f"wandb logging: {self.log_wandb}")

        if "run_name" not in self.config["solver"]["wandb"].keys():
            self.config["solver"]["wandb"]["run_name"] = self.slurm_job_id

        if self.rank == 0 and self.log_wandb:
            os.makedirs(self.config["solver"]["logdir"], exist_ok=True)

            wandb.init(
                project=self.config["solver"]["wandb"]["project_name"],
                name=self.config["solver"]["wandb"]["run_name"],
                dir=self.config["solver"]["logdir"],
                config=self.config,
                mode="offline",
            )
            logger.info(
                f"wandb project name: {self.config['solver']['wandb']['project_name']}"
            )
            logger.info(f"wandb run name: {self.config['solver']['wandb']['run_name']}")
            logger.info(f"wandb dir: {self.config['solver']['logdir']}")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.summary_writer = None
        self.log_file = None
        self.eval_rst = dict()

        self.log_per_iter = self.config["solver"]["log_per_iter"]

        self.accumulation_steps = self.config["solver"]["accumulation_steps"]

        self.global_step = 0

    def get_model(self, config):
        raise NotImplementedError

    def get_dataset(self, config):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def eval_step(self, batch):
        raise NotImplementedError

    def embd_decoder_func(self, i, j, embedding):
        raise NotImplementedError

    def get_embd(self, batch):
        raise NotImplementedError

    def clip_grad_norm(self):
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

    def config_dataloader(self, disable_train_data=False):
        config_train, config_test = (
            self.config["data"]["train"],
            self.config["data"]["test"],
        )

        if not disable_train_data and not config_train["disable"]:
            self.train_loader = self.get_dataloader(config_train)
            self.train_iter = iter(self.train_loader)

        if not config_test["disable"]:
            self.test_loader = self.get_dataloader(config_test)
            self.test_iter = iter(self.test_loader)

    def get_dataloader(self, config):
        dataset, collate_fn = self.get_dataset(config)
        sampler = InfSampler(dataset, shuffle=config["shuffle"])
        data_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        return data_loader

    def config_model(self):
        config = self.config["model"]
        model = self.get_model(config)
        model.to(self.device)
        if self.is_master:
            logger.info(model)
        self.model = model

    def config_optimizer(self):
        config = self.config["solver"]
        parameters = self.model.parameters()

        if config["type"].lower() == "sgd":
            self.optimizer = SGD(
                parameters,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                momentum=0.9,
            )
        elif config["type"].lower() == "adam":
            self.optimizer = Adam(
                parameters, lr=config["lr"], weight_decay=config["weight_decay"]
            )
        elif config["type"].lower() == "adamw":
            self.optimizer = AdamW(
                parameters, lr=config["lr"], weight_decay=config["weight_decay"]
            )
        else:
            raise ValueError(
                "Unknown optimizer type, only support 'sgd', 'adam' and 'adamw'"
            )

    def config_lr_scheduler(self):
        self.scheduler = get_lr_scheduler(self.optimizer, self.config["solver"])

    def configure_log(self, set_writer=True):
        self.logdir = self.config["solver"]["logdir"]
        self.ckpt_dir = os.path.join(self.logdir, "checkpoints")
        self.log_file = os.path.join(self.logdir, "log.csv")

        if self.is_master:
            logging.info("logdir: " + self.logdir)

        if self.is_master and set_writer:
            self.summary_writer = SummaryWriter(self.logdir, flush_secs=20)
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)

    def train_epoch(self, epoch, pbar):
        self.model.train()

        tick = time.time()
        elapsed_time = dict()
        train_tracker = AverageTracker()
        rng = range(len(self.train_loader))

        if self.batch_backprop:
            self.optimizer.zero_grad()

        # if rng is 1, don't use tqdm
        if len(rng) == 1:
            self.disable_tqdm = True

        for it in tqdm(
            range(len(self.train_loader)),
            desc=f"Train Epoch {epoch}",
            position=1,
            leave=False,
            disable=self.disable_tqdm,
        ):
            # load data
            batch = self.train_iter.__next__()
            batch["iter_num"] = it
            batch["epoch"] = epoch
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            elapsed_time["time/data"] = torch.Tensor([time.time() - tick])

            # forward and backward
            output = self.train_step(batch)

            if self.batch_backprop:
                loss = output["train/loss"] / self.accumulation_steps
                loss.backward()
                # apply the gradient every accumulation_steps
                if (self.global_step + 1) % self.accumulation_steps == 0:
                    self.clip_grad_norm()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.accumulation_steps > 1:
                        logging.info(
                            f"Successfully ran accumulated gradient step at step {self.global_step}"
                        )

            # track the averaged tensors
            elapsed_time["time/batch"] = torch.Tensor([time.time() - tick])
            tick = time.time()
            output.update(elapsed_time)
            train_tracker.update(output)

            if (
                it % 50 == 0
                and self.config["solver"]["empty_cache"]
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

            if self.log_per_iter > 0 and self.global_step % self.log_per_iter == 0:
                train_tracker.log(
                    epoch,
                    msg_tag="- ",
                    notes=f"iter: {self.global_step}",
                    print_time=False,
                    pbar=pbar,
                )

            self.global_step += 1

        # Apply gradients if any remain after the loop finishes
        if (self.global_step % self.accumulation_steps != 0) and self.batch_backprop:
            self.clip_grad_norm()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.accumulation_steps > 1:
                logging.info(
                    f"Successfully ran accumulated gradient step at step {self.global_step}"
                )

        if self.rank == 0:
            log_data = train_tracker.average()
            log_data["lr"] = self.optimizer.param_groups[0]["lr"]
            wandb.log(log_data, step=self.global_step)

    def test_epoch(self, epoch, pbar):
        self.model.eval()
        test_tracker = AverageTracker()

        for it in tqdm(
            range(len(self.test_loader)),
            desc=f"Test Epoch {epoch}",
            position=1,  # same inner position as train_epoch
            leave=False,
            disable=self.disable_tqdm,
        ):
            # forward
            batch = self.test_iter.__next__()
            batch["iter_num"] = it
            batch["epoch"] = epoch

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                output = self.test_step(batch)

            # track the averaged tensors
            test_tracker.update(output)

        test_tracker.log(
            epoch, self.summary_writer, self.log_file, msg_tag="=>", pbar=pbar
        )

        if self.rank == 0:
            log_data = test_tracker.average()
            wandb.log(log_data, step=self.global_step)

    def eval_epoch(self, epoch):
        self.model.eval()
        eval_step = min(self.config["solver"]["eval_step"], len(self.test_loader))
        if eval_step < 1:
            eval_step = len(self.test_loader)
        for it in tqdm(range(eval_step), ncols=80, leave=False):
            batch = self.test_iter.__next__()
            batch["iter_num"] = it
            batch["epoch"] = epoch
            with torch.no_grad():
                self.eval_step(batch)

    def save_checkpoint(self, epoch):
        # save checkpoint
        model_dict = self.model.state_dict()
        ckpt_name = os.path.join(self.ckpt_dir, "%05d" % epoch)
        torch.save(model_dict, ckpt_name + ".model.pth")
        torch.save(
            {
                "model_dict": model_dict,
                "epoch": epoch,
                "optimizer_dict": self.optimizer.state_dict(),
                "scheduler_dict": self.scheduler.state_dict(),
            },
            ckpt_name + ".solver.tar",
        )
        logging.info(f"Checkpoint saved to {ckpt_name}")

    def load_checkpoint(self):
        ckpt = self.config["solver"]["ckpt"]
        if (not ckpt) or (ckpt is None):
            return
        map_location = self.device  # Ensure checkpoint loads on correct device
        trained_dict = torch.load(ckpt, map_location=map_location)
        model_dict = trained_dict.get("model_dict", trained_dict)
        self.model.load_state_dict(model_dict)
        if "epoch" in trained_dict:
            self.start_epoch = trained_dict["epoch"] + 1
        if "optimizer_dict" in trained_dict and self.optimizer:
            self.optimizer.load_state_dict(trained_dict["optimizer_dict"])
        if "scheduler_dict" in trained_dict and self.scheduler:
            self.scheduler.load_state_dict(trained_dict["scheduler_dict"])

    def manual_seed(self):
        rand_seed = self.config["solver"]["rand_seed"]
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(rand_seed)
                torch.cuda.manual_seed_all(rand_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_lr_scheduler()
        self.configure_log()
        self.load_checkpoint()

        rng = range(self.start_epoch, self.config["solver"]["max_epoch"] + 1)
        with tqdm(
            rng, desc="Epoch", position=0, leave=True, disable=self.disable_tqdm
        ) as pbar:
            for epoch in pbar:
                # training epoch
                self.train_epoch(epoch, pbar)

                # update learning rate
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()
                self.summary_writer.add_scalar("train/lr", lr[0], epoch)

                # testing epoch
                # if epoch % self.config["solver"]["test_every_epoch"] == 0:
                # self.test_epoch(epoch, pbar)

                # checkpoint
                if epoch % self.config["solver"]["save_every_epoch"] == 0:
                    self.save_checkpoint(epoch)

    def test(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.test_epoch(epoch=0)

    def evaluate(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        for epoch in tqdm(range(self.config["solver"]["eval_epoch"]), ncols=80):
            self.eval_epoch(epoch)

    def profile(self):
        r"""Set `DATA.train.num_workers 0` when using this function."""

        # Ensure model and dataloader are configured
        self.config_model()
        self.config_dataloader()

        logdir = self.config["solver"]["logdir"]

        # Check PyTorch version
        version = torch.__version__.split(".")
        larger_than_110 = int(version[0]) > 0 and int(version[1]) > 10
        if not larger_than_110:
            logger.info("The profile function is only available for Pytorch>=1.10.0.")
            return

        # Warm-up phase to ensure initial profiling works well
        batch = next(iter(self.train_loader))
        batch = {
            k: v.to(self.device) for k, v in batch.items()
        }  # Ensure data is on the right device
        for _ in range(3):
            output = self.train_step(batch)
            output["train/loss"].backward()

        # Start profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:

            for i in range(3):
                output = self.train_step(batch)
                output["train/loss"].backward()
                prof.step()

        # Log the profiling results, sorted by GPU time and memory usage
        logger.info(
            prof.key_averages(group_by_input_shape=True, group_by_stack_n=10).table(
                sort_by="cuda_time_total", row_limit=10
            )
        )
        logger.info(
            prof.key_averages(group_by_input_shape=True, group_by_stack_n=10).table(
                sort_by="cuda_memory_usage", row_limit=10
            )
        )

    def run(self):
        eval("self.%s()" % self.config["solver"]["run"])
        wandb.finish()
