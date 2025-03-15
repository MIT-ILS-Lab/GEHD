import torch
import os
import shutil
import sys
import numpy as np

from pyvrp import Solution


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith("ipykernel_launcher"):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, "src")

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, "__file__") and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, "({})")
                    filepath = "".join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                try:
                    shutil.copy(src_abspath, dst_filepath)
                except:
                    print("Failed to copy file: {}".format(src_abspath))


def cumsum(data: torch.Tensor, dim: int, exclusive: bool = False):
    r"""Extends :func:`torch.cumsum` with the input argument :attr:`exclusive`.

    Args:
      data (torch.Tensor): The input data.
      dim (int): The dimension to do the operation over.
      exclusive (bool): If false, the behavior is the same as :func:`torch.cumsum`;
          if true, returns the cumulative sum exclusively. Note that if ture,
          the shape of output tensor is larger by 1 than :attr:`data` in the
          dimension where the computation occurs.
    """

    out = torch.cumsum(data, dim)

    if exclusive:
        size = list(data.size())
        size[dim] = 1
        zeros = out.new_zeros(size)
        out = torch.cat([zeros, out], dim)
    return out


def parse_pyvrp_solution(pyvrp_solution: Solution):
    """
    Parse a PyVRP solution object into a dictionary.
    """
    distance = pyvrp_solution.distance()
    is_feasible = pyvrp_solution.is_feasible()
    num_routes = pyvrp_solution.num_routes()
    routes = pyvrp_solution.routes()
    solution = [0]
    for route in routes:
        solution += route.visits()
        solution.append(0)
    return {
        "distance": distance,
        "is_feasible": is_feasible,
        "num_routes": num_routes,
        "solution": solution,
    }
