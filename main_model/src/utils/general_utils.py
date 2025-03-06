import torch


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
