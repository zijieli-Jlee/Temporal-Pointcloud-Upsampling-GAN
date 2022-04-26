import numpy as np
import numba as nb


@nb.jit
def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=-1)


@nb.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result


@nb.njit
def np_argmax(array, axis):
  return np_apply_along_axis(np.argmax, axis, array)


@nb.njit
def sample_loop(indices, min_distances, distances, pts, k):
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        farthest_point = pts[indices[i]]
        dist = l2_norm(farthest_point.reshape(1, -1), pts)
        distances[i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return


# this part of code is modified and copied from:
# https://github.com/corochann/chainer-pointnet/blob/master/chainer_pointnet/utils/sampling.py
# which works for both cpu and gpu
def farthest_point_sampling(pts, k, initial_idx=None,
                            skip_initial=False, indices_dtype=np.int64,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """

    assert pts.ndim == 2
    num_point, coord_dim = pts.shape
    indices = np.zeros((k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = np.zeros((k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[0] = np.random.randint(len(pts))
    else:
        indices[0] = initial_idx

    farthest_point = pts[indices[0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = l2_norm(farthest_point[None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()
    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[0] = np.argmax(min_distances, axis=1)
        farthest_point = pts[indices[0]]
        min_distances = l2_norm(farthest_point[None, :], pts)

    distances[0, :] = min_distances
    sample_loop(indices, min_distances, distances, pts, k)
    return indices, distances
