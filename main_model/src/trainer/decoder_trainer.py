import time
import wandb
import logging
import h5py

import polyscope as ps
import trimesh
import numpy as np
import pygeodesic

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


class LEHDTrainer(Solver):
    def __init__(self, config, is_master=True):
        super().__init__(config, is_master)

        # Store trainer and testing params
        self.mesh_params = config["data"]["mesh"]
        self.trainer_params = config["data"]["train"]
        self.testing_params = config["data"]["test"]

        self.city_indices = None

        # Set random seed
        torch.manual_seed(22)

    def get_model(self, config):
        # Create and return the LEHD model
        return LEHD(self.city_indices, **config).to(self.device)

    def get_dataset(self, config):
        return get_dataset(config)

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
            self.test_iter = {
                key: iter(loader) for key, loader in self.test_loader.items()
            }

    def get_dataloader(self, config):
        # Override to use custom batch sampler
        dataset, collate_fn = self.get_dataset(config)

        # Determine if this is train or test config
        is_train = config["mode"] == "train"
        params = self.trainer_params if is_train else self.testing_params

        if not is_train:
            dataloaders = {}
            for size, sub_dataset in dataset.datasets.items():
                dataset_size = len(sub_dataset)
                problem_size = sub_dataset.raw_data_problems.shape[1] - 1
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
                    sub_dataset,
                    batch_sampler=infinite_batch_sampler,
                    collate_fn=collate_fn,
                    num_workers=params["num_workers"],
                    pin_memory=True,
                    persistent_workers=True,
                )
                dataloaders[size] = data_loader

            return dataloaders
        else:
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

    def config_mesh(self):
        """Load the mesh data to get node coordinates"""
        with h5py.File(self.mesh_params["mesh_path"], "r") as hf:
            # Load mesh data
            self.vertices = torch.tensor(hf["vertices"][:], requires_grad=False)
            self.faces = torch.tensor(hf["faces"][:], requires_grad=False)
            self.city = torch.tensor(hf["city"][:], requires_grad=False)
            self.city_indices = torch.tensor(hf["city_indices"][:], requires_grad=False)
            self.geodesic_matrix = torch.tensor(
                hf["geodesic_matrix"][:], requires_grad=False
            ).to(self.device)

        logging.info(f"Loaded mesh data with {len(self.city)} city points")

    def train(self):
        self.manual_seed()
        self.config_mesh()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_lr_scheduler()
        self.configure_log()
        self.load_checkpoint()

        self.time_estimator.reset(self.start_epoch)

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

    def test(self):
        self.manual_seed()
        self.config_mesh()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.test_epoch(epoch=0)

    def visualize(self):
        self.manual_seed()
        self.config_mesh()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()

        assert (
            self.config["solver"]["ckpt"] is not None
        ), "Checkpoint path in ckpt need to be provided."

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

            elapsed_time["train/time/data"] = torch.Tensor([time.time() - tick])

            # forward and backward
            output = self.train_step(batch)

            # track the averaged tensors
            elapsed_time["train/time/batch"] = torch.Tensor([time.time() - tick])
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
                    "Epoch {:3d}: Train {:3d}/{:3d} ({:5.1f}%) Loss: {:.4f} Feasibility Loss: {:.4f} Combined: {:.4f} Time: {:.2f}".format(
                        epoch,
                        episode,
                        len(self.train_loader),
                        episode / len(self.train_loader) * 100,
                        output["train/loss"],
                        output["train/feasibility_loss"],
                        output["train/combined_loss"],
                        output["train/time/batch"].item() / 60,
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
            "Avg. Loss: {:.2f} Avg. Feasibility Loss: {:.2f} Avg. Combined Loss: {:.2f} Avg. Time: {:.2f} min".format(
                train_tracker_epoch.average()["train/loss"],
                train_tracker_epoch.average()["train/feasibility_loss"],
                train_tracker_epoch.average()["train/combined_loss"],
                train_tracker_epoch.average()["train/time/batch"] / 60,
            )
        )

    def test_epoch(self, epoch):
        self.model.eval()

        for key in self.test_loader.keys():

            # log which key we are currently testing
            logger.info(f"*** Testing instance sice {key} ***")

            test_tracker = AverageTracker()
            tick = time.time()  # Start time for batch timing
            elapsed_time = dict()

            for episode in range(
                1, len(self.test_loader[key]) + 1
            ):  # Simple loop without tqdm
                # Load data
                batch = self.test_iter[key].__next__()
                batch["iter_num"] = episode
                batch["epoch"] = epoch

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                elapsed_time[f"test/{key}/time/data"] = torch.Tensor(
                    [time.time() - tick]
                )

                # Forward pass using test_step
                with torch.no_grad():
                    output = self.test_step(batch, key)

                elapsed_time[f"test/{key}/time/batch"] = torch.Tensor(
                    [time.time() - tick]
                )
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
                            output[f"test/{key}/gap_percentage"],
                            output[f"test/{key}/time/batch"].item() / 60,
                        )
                    )

            # Log final averaged metrics to wandb and console
            # Logg averages
            logger.info(" ")
            logger.info("*** Summary ***")
            logger.info(
                "Avg. Opt. Score: {:.2f}, Avg. St. Score: {:.2f}".format(
                    test_tracker.average()[f"test/{key}/optimal_score"],
                    test_tracker.average()[f"test/{key}/student_score"],
                )
            )
            logger.info(
                "Avg. Gap: {:.2f}%, Avg. Time {:.2f} min".format(
                    test_tracker.average()[f"test/{key}/gap_percentage"],
                    test_tracker.average()[f"test/{key}/time/batch"] / 60,
                )
            )
            logger.info(" ")

            if self.rank == 0:
                log_data = test_tracker.average()
                wandb.log(log_data, step=self.global_step)

    def train_step(self, batch):
        # Extract data from batch
        solutions = batch["solutions"]
        capacities = batch["capacities"].float()

        loss_list = []
        feasibility_loss_list = []

        # Training loop for constructing solution
        while (
            solutions.size(1) > 2
        ):  # if solutions.size(1) == 3, only start, destination and depot left
            logits, feasibility_logits = self.model(
                solutions,
                capacities,
            )

            # Main routing loss
            flag_teacher = solutions[:, 1, -1].to(torch.int64)

            indices_flag = torch.roll(
                solutions[:, 1:-1, -1].to(torch.bool), shifts=-1, dims=1
            )
            indices_flag += solutions[:, 1:-1, -1].to(torch.bool)
            indices_flag = indices_flag.to(torch.int64)

            indices_teacher = torch.zeros(logits.shape).to(self.device)
            indices_teacher[:, 0] = 1.0 - flag_teacher
            indices_teacher[:, (solutions.size(1) - 2) :] = (
                indices_flag * flag_teacher[:, None]
            )

            indices_teacher = indices_teacher / indices_teacher.sum(dim=1, keepdim=True)

            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, indices_teacher, reduction="batchmean")

            # loss = F.binary_cross_entropy_with_logits(logits, indices_teacher)

            # Feasibility loss - create binary labels based on capacity constraints
            remaining_capacity = solutions[:, 0, 3]
            demands = solutions[:, 1:-1, 2]
            feasibility_labels = (demands <= remaining_capacity.unsqueeze(1)).float()
            feasibility_loss = F.binary_cross_entropy_with_logits(
                feasibility_logits, feasibility_labels
            )

            # Combined loss with weighting
            combined_loss = loss + feasibility_loss  # You can adjust the weight

            # Backpropagate and update model
            self.model.zero_grad()
            combined_loss.backward()

            self.clip_grad_norm()
            self.optimizer.step()

            # Update capacity in problems tensor directly
            # 1. If flag = 1, the vehicle returns to depot and capacity is refilled
            is_depot = flag_teacher == 1
            solutions[is_depot, :, 3] = capacities[is_depot, None]

            # 2. Get demands of selected nodes using gather
            selected_demands = solutions[:, 1, 2]

            # 3. If capacity is less than demand, capacity is refilled and flag is changed to 1
            smaller_ = solutions[:, 0, 3] < selected_demands
            solutions[smaller_, :, 3] = capacities[smaller_, None]

            # 4. Subtract demand from capacity
            solutions[:, :, 3] = solutions[:, :, 3] - selected_demands[:, None]

            # 5. Update problems tensor for next step
            solutions = solutions[:, 1:, :]

            loss_list.append(loss)
            feasibility_loss_list.append(feasibility_loss)

        # Calculate final loss
        loss_mean = torch.stack(loss_list).mean()
        feasibility_loss_mean = torch.stack(feasibility_loss_list).mean()

        # Return output dictionary for tracker
        return {
            "train/loss": loss_mean,
            "train/feasibility_loss": feasibility_loss_mean,
            "train/combined_loss": loss_mean + 0.5 * feasibility_loss_mean,
        }

    def test_step(self, batch, key: None, eval: bool = False):
        # Extract data from batch
        solutions = batch["solutions"]
        capacities = batch["capacities"].float()
        costs = batch["costs"].float()

        batch_size = solutions.size(0)

        solutions_orig = solutions.clone()

        selected_student_list = torch.empty(
            (solutions.size(0), 0), device=solutions.device
        )
        selected_student_flag = torch.empty(
            (solutions.size(0), 0), device=solutions.device, dtype=torch.int64
        )
        # Training loop for constructing solution
        first_step = True
        while (
            solutions.size(1) > 2
        ):  # if solutions.size(1) == 3, only start, destination and depot left
            logits, _ = self.model(
                solutions,
                capacities,
            )

            if not first_step:
                remaining_capacity = solutions[:, 0, 3]
                demands = solutions[:, 1:-1, 2]
                feasibility_labels = demands <= remaining_capacity.unsqueeze(1)
                logits[:, : solutions.size(1) - 2][~feasibility_labels] = -float("inf")
            else:
                logits[:, : solutions.size(1) - 2] = -float("inf")
                first_step = False

            indices = logits.argmax(dim=1)

            assert logits.shape[1] == 2 * (
                solutions.size(1) - 2
            ), "Logits shape mismatch"

            flag_student = (indices >= solutions.size(1) - 2).to(torch.int64)
            node_indices = indices - flag_student * (solutions.size(1) - 2)

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

            # assert smaller_.sum().item() == 0, "Capacity smaller than demand"

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

        depots = solutions_orig[:, -1, 0].unsqueeze(1)

        # Calculate optimal and student scores
        optimal_length = self.get_travel_distance(
            solutions_orig[:, 1:-1, 0]
            .clone()
            .to(torch.int64),  # exclude depot from the solution vector
            solutions_orig[:, 1:-1, 4]
            .clone()
            .to(torch.int64),  # exclude depot from the solution vector
            depots,
        )
        current_best_length = self.get_travel_distance(
            selected_student_list.clone().to(torch.int64),
            selected_student_flag.clone().to(torch.int64),
            depots,
        )

        # assert (optimal_length - costs).mean() < 1e-1, "Optimal length mismatch"

        # Calculate gap as percentage
        gap = 100 * ((current_best_length - optimal_length) / optimal_length).mean()

        if not eval:
            return {
                f"test/{key}/optimal_score": optimal_length.mean(),
                f"test/{key}/student_score": current_best_length.mean(),
                f"test/{key}/gap_percentage": gap,
            }
        else:
            return {
                f"test/{key}/optimal_score": optimal_length.mean(),
                f"test/{key}/student_score": current_best_length.mean(),
                f"test/{key}/gap_percentage": gap,
                f"test/{key}/optimal_solution/nodes": solutions_orig[:, 1:-1, 0],
                f"test/{key}/optimal_solution/flags": solutions_orig[:, 1:-1, 4],
                f"test/{key}/student_solution/nodes": selected_student_list,
                f"test/{key}/student_solution/flags": selected_student_flag,
                f"test/{key}/depots": depots,
            }

    def get_travel_distance(self, order_node, order_flag, depots):
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = depots.expand_as(order_flag_)[index_bigger].to(
            torch.int64
        )

        roll_node = order_node_.roll(dims=1, shifts=1)

        order_loc = order_node_
        roll_loc = roll_node
        flag_loc = order_flag_

        order_lengths = self.geodesic_matrix[order_loc, flag_loc]

        order_flag_[:, 0] = depots.squeeze(-1)

        flag_loc = order_flag_

        roll_lengths = self.geodesic_matrix[roll_loc, flag_loc]

        length = order_lengths.sum(dim=1) + roll_lengths.sum(dim=1)

        return length

    def visualize_single_solution(self, nodes, flags, depot):
        node_sequence = []
        for flag, node in zip(flags, nodes):
            if flag.item() == 1:
                node_sequence.append(int(depot.item()))
            node_sequence.append(int(node.item()))
        node_sequence.append(int(depot.item()))
        node_indices = self.city_indices[node_sequence].tolist()
        depot_index = self.city_indices[int(depot.item())].tolist()

        # Initialize Polyscope
        ps.init()
        ps.set_up_dir("z_up")
        ps.remove_all_structures()

        # Create Trimesh object
        mesh = trimesh.Trimesh(vertices=self.vertices.numpy(), faces=self.faces.numpy())

        # Initialize geodesic algorithm
        geoalg = pygeodesic.geodesic.PyGeodesicAlgorithmExact(
            self.vertices.numpy(), self.faces.numpy()
        )

        # Register mesh geometry in Polyscope
        ps_mesh = ps.register_surface_mesh(
            "Solution Mesh",
            self.vertices.numpy(),
            self.faces.numpy(),
            color=(0.8, 0.8, 0.8),
            transparency=0.7,
        )

        # Split solution into individual routes
        routes = []
        start_idx = 0
        for i in range(1, len(node_indices)):
            if node_indices[i] == depot_index:
                routes.append(node_indices[start_idx : i + 1])
                start_idx = i

        # Assign route color to each node
        colors = plt.cm.tab10.colors
        node_to_route_color = {}
        for route_idx, route in enumerate(routes):
            route_color = colors[route_idx % len(colors)]
            for n in route:
                node_to_route_color[n] = route_color

        # Visualize routes
        for route_idx, route in enumerate(routes):
            route_color = colors[route_idx % len(colors)]
            full_path = []
            for i in range(len(route) - 1):
                src, dst = route[i], route[i + 1]
                path_points = self._get_geodesic_path(geoalg, src, dst)
                if path_points is not None:
                    full_path.extend(path_points)
            if len(full_path) > 1:
                self.visualize_geodesic_path(
                    np.array(full_path), route_color, route_idx
                )

        # Visualize nodes: smaller and colored by route
        all_nodes = np.unique(np.concatenate(routes))
        node_coords = self.vertices.numpy()[all_nodes]
        node_colors = []
        for node_idx, node in enumerate(all_nodes):
            if node == depot_index:
                node_colors.append([1.0, 0.0, 0.0])  # Red for depot
            else:
                # Fetch route color, default to gray if not found
                route_color = node_to_route_color.get(node, [0.5, 0.5, 0.5])
                node_colors.append(route_color)
        node_colors = np.array(node_colors)

        ps_nodes = ps.register_point_cloud(
            "Nodes", node_coords, radius=0.015, enabled=True  # << SMALLER NODES!
        )
        ps_nodes.add_color_quantity("Route color", node_colors, enabled=True)

        ps.show()

    def _get_geodesic_path(self, geoalg, src_idx, dst_idx):
        """Calculate geodesic distance between two nodes using pygeodesic"""
        source_indices = np.array([src_idx], dtype=np.int32)
        target_indices = np.array([dst_idx], dtype=np.int32)

        # Calculate geodesic distance using pygeodesic
        _, path = geoalg.geodesicDistance(target_indices, source_indices)
        return path

    def visualize_geodesic_path(self, geodesic_path, route_color, route_idx):
        """Visualizes a geodesic path on a mesh using Polyscope.

        Args:
            geodesic_path (np.array): A numpy array of shape (N, 3) representing the 3D coordinates of the geodesic path.
            route_color (tuple): A tuple of three floats representing the RGB color of the route (e.g., (1, 0, 0) for red).
        """
        if geodesic_path is None or len(geodesic_path) < 2:
            print("Cannot visualize: Invalid or empty geodesic path.")
            return

        # Register the curve network with Polyscope
        ps_curve = ps.register_curve_network(
            f"Geodesic Path: {route_idx}",
            geodesic_path,
            edges=np.array([[i, i + 1] for i in range(len(geodesic_path) - 1)]),
            color=route_color,
            radius=0.005,
        )

    def visualize_solutions(self, keys=None):
        self.model.eval()

        keys = keys if keys is not None else self.test_loader.keys()

        for key in keys:
            assert key in self.test_loader.keys(), f"Key {key} not found in test_loader"

        for key in keys:
            test_tracker = AverageTracker()
            tick = time.time()  # Start time for batch timing
            elapsed_time = dict()

            batch = self.test_iter[key].__next__()

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            elapsed_time[f"test/{key}/time/data"] = torch.Tensor([time.time() - tick])

            # Forward pass using test_step
            with torch.no_grad():
                output = self.test_step(batch, key, eval=True)

            elapsed_time[f"test/{key}/time/batch"] = torch.Tensor([time.time() - tick])
            tick = time.time()

            # Update tracker with metrics and timing
            output.update(elapsed_time)
            test_tracker.update(output)

            optimal_solution_nodes = output[f"test/{key}/optimal_solution/nodes"]
            optimal_solution_flags = output[f"test/{key}/optimal_solution/flags"]
            student_solution_nodes = output[f"test/{key}/student_solution/nodes"]
            student_solution_flags = output[f"test/{key}/student_solution/flags"]
            depots = output[f"test/{key}/depots"]

            for i in range(optimal_solution_nodes.shape[0]):
                optimal_nodes = optimal_solution_nodes[i]
                optimal_flags = optimal_solution_flags[i]
                student_nodes = student_solution_nodes[i]
                student_flags = student_solution_flags[i]

                depot = depots[i]

                # Visualize the solutions
                self.visualize_single_solution(optimal_nodes, optimal_flags, depot)
                self.visualize_single_solution(student_nodes, student_flags, depot)
