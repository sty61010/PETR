from typing import Callable, Dict, Optional, Tuple, Union
import enum
import torch


class NormalizeMode(enum.Enum):
    ZERO_TO_ONE = enum.auto()
    MINUS_ONE_TO_ONE = enum.auto()


def normalize_to_0_1(uv: torch.Tensor, depth: torch.Tensor, img_shape: Tuple[int]) -> Tuple[torch.Tensor, ...]:
    img_H, img_W, _ = img_shape
    normalized_uv = uv / uv.new_tensor([img_W, img_H]).expand_as(uv)

    # [N_0, N_1, ..., 2]
    mask = (torch.isfinite(normalized_uv)
            & (normalized_uv[..., 0:1] >= 0.0)
            & (normalized_uv[..., 0:1] <= 1.0)
            & (normalized_uv[..., 1:2] >= 0.0)
            & (normalized_uv[..., 1:2] <= 1.0)
            & (depth >= 0.0))
    # [N_0, N_1, ...]
    mask = torch.all(mask, dim=-1)

    return normalized_uv, mask


def normalize_to_minus_1_1(uv: torch.Tensor, depth: torch.Tensor, img_shape: Tuple[int]) -> Tuple[torch.Tensor, ...]:
    img_H, img_W, _ = img_shape
    normalized_uv = uv / uv.new_tensor([img_W, img_H]).expand_as(uv)

    # [N_0, N_1, ..., 2]
    mask = (torch.isfinite(normalized_uv)
            & (normalized_uv[..., 0:1] >= -1.0)
            & (normalized_uv[..., 0:1] <= 1.0)
            & (normalized_uv[..., 1:2] >= -1.0)
            & (normalized_uv[..., 1:2] <= 1.0)
            & (depth >= 0.0))
    # [N_0, N_1, ...]
    mask = torch.all(mask, dim=-1)

    return normalized_uv, mask


_NORMALIZE_FUNCTIONS = {
    NormalizeMode.ZERO_TO_ONE: normalize_to_0_1,
    NormalizeMode.MINUS_ONE_TO_ONE: normalize_to_minus_1_1,
}


def convert_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    homogeneous_points = torch.cat([
        points,
        points.new_ones((*points.shape[:-1], 1))
    ], dim=-1)

    return homogeneous_points


def project_ego_to_image(points: torch.Tensor,
                         lidar2img: torch.Tensor,
                         img_shape: Optional[Tuple[int]] = None,
                         return_depth: bool = False,
                         return_normalized_uv: bool = False,
                         normalize_method: Optional[Union[NormalizeMode,
                                                          Callable[[torch.Tensor, Tuple[int]],
                                                                   Tuple[torch.Tensor, ...]]]] = None,
                         return_batch_first: bool = True,
                         ) -> Dict[str, torch.Tensor]:
    """Project ego-pose coordinate to image coordinate

    Args:
        * points: 3D (x, y, z) reference points in ego-pose
            with shape `[B, num_queries, num_levels, 3]`.
        * lidar2img: Transform matrix from lidar (ego-pose) coordinate to
            image coordinate with shape `[B, num_cameras, 4, 4]` or `[B, num_cameras, 3, 4]`.
        * img_shape: Image shape (height, width, channel).
            Must be specified if `normalize_method` is not None nor Callable.
            Note that this is not the input shape of the frustum.
            This is the shape with respect to the intrinsic.
        * return_depth: Whether to return depth for each points in each cameras.
        * return_normalized_uv: Whether to return normalized `uv` points.
        * normalize_method: Methods to normalize `uv` points.
            It can be a member of `NormalizeMode` enum or any user-defined Callable taking
            `uv`, `depth`, and `img_shape` as inputs and output the normalized `uv` and `mask`.
        * return_batch_first: Whether to return batch first
            (`[B, num_cameras, ...]`) or camera first (`[num_cameras, B, ...]`)
    Returns:
        * uv: The projected points (u, v) of each camera with shape
            [B, num_cameras, num_queries, num_levels, 2].
        * mask: The mask of the valid projected points with shape
            [B, num_cameras, num_queries, num_levels].
        * (Optional) depth: The depth of each points in each camera coordinate with shape
            [B, num_cameras, num_queries, num_levels, 1].
            Returns only if `return_depth` is True.
            It may contains invalid values, only the positions where `mask[n, b, q, lvl]` are True are valid.
        * (Optional) normalized_uv: The normalized projected points (u, v) of each camera
            with shape [B, num_cameras, num_queries, num_levels, 2].
            All elements are normalized if `normalize_method` is specified.

            If `normalize_method` == '0_to_1':
                All elements are in the range of `[0, 1]` (top-left: `(0, 0)`, bottom-right: `(1, 1)`).
            If `normalize_method` == '-1_to_1':
                All elements are in the range of `[-1, 1]` (top-left: `(-1, -1)`, bottom-right: `(1, 1)`).
            If `normalize_method` is a Callable, `uv` will be normalized by `uv = normalize_method(uv)`.

    """
    assert points.shape[0] == lidar2img.shape[0], (
        f'The number in the batch dimension must be equal. points: {points.shape}, lidar2img: {lidar2img.shape}')
    B, num_queries, num_levels, _ = points.shape
    _, num_cameras, _, _ = lidar2img.shape

    lidar2img = lidar2img[:, :, :3]
    # convert to homogeneous coordinate. [batch, num_queries, num_levels, 4]
    points = convert_to_homogeneous(points)

    # [batch, num_cameras, num_queries, num_levels, 3]
    uvd: torch.Tensor = torch.einsum('bnij,bqlj->bnqli', lidar2img, points)
    if not return_batch_first:
        # [num_cameras, batch, num_queries, num_levels, 3]
        uvd = uvd.transpose(0, 1)
    uv, depth = uvd[..., :2], uvd[..., 2:3]
    # [batch, num_cameras, num_queries, num_levels, 2]. Must not be in-place because `uv` is a view.
    uv = uv / depth

    if return_normalized_uv:
        assert normalize_method is not None, '`normalized_method` must not be None if `return_normalized_uv` is True'
        assert img_shape is not None, '`img_shape` must not be None if `return_normalized_uv` is True'

        if isinstance(normalize_method, NormalizeMode):
            normalize_method = _NORMALIZE_FUNCTIONS[normalize_method]
        # uv: [batch, num_cameras, num_queries, num_levels, 2]
        # mask: [batch, num_cameras, num_queries, num_levels]
        normalized_uv, mask = normalize_method(uv, depth, img_shape)
        if return_batch_first:
            assert normalized_uv.shape == (B, num_cameras, num_queries, num_levels, 2)
            assert mask.shape == (B, num_cameras, num_queries, num_levels)
        else:
            assert normalized_uv.shape == (num_cameras, B, num_queries, num_levels, 2)
            assert mask.shape == (num_cameras, B, num_queries, num_levels)
    else:
        # [batch, num_cameras, num_queries, num_levels]
        mask = (torch.isfinite(uv) & depth >= 0.0).all(dim=-1)

    uv = uv.masked_fill(~mask.unsqueeze(-1), 0)
    depth = depth.masked_fill(~mask.unsqueeze(-1), 0)
    ret_dict = dict(uv=uv, mask=mask)
    if return_depth:
        ret_dict['depth'] = depth
    if return_normalized_uv:
        ret_dict['normalized_uv'] = normalized_uv
    return ret_dict
