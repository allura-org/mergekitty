# Copyright (C) 2025 Allura-org
# `slerp`, `fft_transform`, `ifft_transform`, `normalize_tensor`, `interpolate_fft_components`, `merge_tensors_fft2_slerp`
# Copyright (C) 2024 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#

# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from typing import List, Optional

import torch

from mergekitty.merge_methods.easy_define import merge_method

logger = logging.getLogger(__name__)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two vectors.

    Parameters:
    - v0 (torch.Tensor): The first vector.
    - v1 (torch.Tensor): The second vector.
    - t (float): Interpolation parameter (0 <= t <= 1).

    Returns:
    - torch.Tensor: Interpolated vector.
    """
    dot = torch.sum(v0 * v1) / (v0.norm() * v1.norm())
    dot = torch.clamp(dot, -1.0, 1.0)  # Clamp to avoid numerical issues

    theta = torch.acos(dot) * t
    relative_vec = v1 - v0 * dot
    relative_vec = torch.nn.functional.normalize(relative_vec, dim=-1)

    return v0 * torch.cos(theta) + relative_vec * torch.sin(theta)


def fft_transform(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Perform FFT on the tensor, considering its dimensionality.

    Parameters:
    - tensor (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: FFT of the input tensor.
    """
    if tensor.ndim == 1:
        return torch.fft.fft(tensor.to(device).to(torch.float32)).to("cpu")
    else:
        return torch.fft.fftn(tensor.to(device).to(torch.float32), dim=(-2, -1)).to(
            "cpu"
        )


def ifft_transform(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Perform inverse FFT on the tensor, considering its dimensionality.

    Parameters:
    - tensor (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Inverse FFT of the input tensor.
    """
    if tensor.ndim == 1:
        return torch.fft.ifft(tensor.to(device)).real.to("cpu")
    else:
        return torch.fft.ifftn(tensor.to(device), dim=(-2, -1)).real.to("cpu")


def normalize_tensor(tensor: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    """
     Normalize a tensor by its norm.
    <|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:
     Parameters:
     - tensor (torch.Tensor): Input tensor.

     Returns:
     - torch.Tensor: Normalized tensor.
    """
    norm_t = tensor.to(device=device).norm()
    norm = norm_t.item()
    del norm_t
    return tensor / norm if norm != 0 else tensor, norm


def interpolate_fft_components(
    v0_fft: torch.Tensor,
    v1_fft: torch.Tensor,
    t: float,
    device: str,
    t_sum: float = 1.0,
    cutoff_pct: float = 0.0,
    cull_pct: float = 0.0,
    interp_imag: bool = True,
) -> torch.Tensor:
    """
    Interpolate the real and imaginary parts of FFT components using SLERP.

    Parameters:
    - v0_fft (torch.Tensor): FFT of the first tensor.
    - v1_fft (torch.Tensor): FFT of the second tensor.
    - t (float): Interpolation parameter (0 <= t <= 1).
    - t_sum (float): Sum of the interpolation parameters.
    - cutoff_pct (float): Floor percentage for doing task addition instead of slerp.
    - cull_pct (float): Floor percentage of the FFT components to cull.
    - interp_imag (bool): Whether to interpolate the imaginary parts of the FFT components.

    Returns:
    - torch.Tensor: Interpolated FFT components.
    """
    result_fft = torch.zeros_like(v0_fft, device=device)

    real_v0 = v0_fft.real.to(device)
    real_v1 = v1_fft.real.to(device)
    abs_real_v0 = real_v0.abs()
    abs_real_v1 = real_v1.abs()

    if cutoff_pct > 0:
        all_real, rest = torch.sort(
            torch.cat([abs_real_v0, abs_real_v1]).ravel(), descending=False
        )
        pct_idx = int(len(all_real) * cutoff_pct)
        if pct_idx >= len(all_real):
            cutoff_threshold = all_real[-1].item()
        else:
            cutoff_threshold = all_real[pct_idx].item()
        del all_real, rest
    else:
        cutoff_threshold = 0

    sign_mask = real_v0.sign() == real_v1.sign()
    small_values_v0 = abs_real_v1 < cutoff_threshold
    small_values_v1 = abs_real_v1 < cutoff_threshold
    slerp_mask = sign_mask & ~small_values_v0 & ~small_values_v1
    sum_mask = sign_mask & ~slerp_mask
    rest_mask = ~slerp_mask & ~sum_mask
    del small_values_v0, small_values_v1, sign_mask
    larger_values_mask = abs_real_v0 > abs_real_v1

    # Interpolate real parts using SLERP
    result_fft.real[slerp_mask] = slerp(real_v0[slerp_mask], real_v1[slerp_mask], t)
    result_fft.real[sum_mask] = real_v0[sum_mask] + t_sum * real_v1[sum_mask]
    result_fft.real[rest_mask] = torch.where(
        larger_values_mask[rest_mask], real_v0[rest_mask], real_v1[rest_mask]
    )

    if cull_pct > 0:
        all_real, rest = torch.sort(result_fft.real.abs().ravel(), descending=False)
        cull_idx = int(len(all_real) * cull_pct)
        cull_threshold = all_real[cull_idx].item()
        # Check to make sure that the cull threshold doesn't take more than it should
        if (all_real < cull_threshold).sum() > (len(all_real) * (cull_pct * 2)):
            logger.info(
                f"Warning: Cull threshold overflow {cull_threshold} {cull_idx} {len(all_real)} {(all_real < cull_threshold).sum()} {len(all_real) * (cull_pct + 0.01)}"
            )
        else:
            logger.debug(
                f"Cull {cull_threshold} {cull_idx} {len(all_real)} {(all_real < cull_threshold).sum()} {len(all_real) * (cull_pct + 0.01)}"
            )
            result_fft.real[torch.abs(result_fft.real) < cull_threshold] = 0
        del all_real, rest

    del (
        real_v0,
        real_v1,
        abs_real_v0,
        abs_real_v1,
        larger_values_mask,
        slerp_mask,
        sum_mask,
        rest_mask,
    )

    if interp_imag:
        i0_fft = fft_transform(v0_fft.imag, device=device)
        i1_fft = fft_transform(v1_fft.imag, device=device)
        # The zero cutoff is more about memory issue then for a good reason
        i0_fft = interpolate_fft_components(
            i0_fft,
            i1_fft,
            t=t,
            cutoff_pct=0,
            cull_pct=0,
            interp_imag=False,
            device=device,
        )
        del i1_fft
        result_fft.imag = ifft_transform(i0_fft, device=device)
    else:
        result_fft.imag = v0_fft.imag

    return result_fft


def merge_tensors_fft2_slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float,
    device: str,
    b: float = 0.1,
    t_sum: float = 1.0,
    cutoff_pct: float = 0.0,
    cull_pct: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """
    Merges two tensors using 2D Fourier transform interpolation with SLERP for both
    the real and imaginary parts.

    Parameters:
    - v0 (torch.Tensor): The first input tensor.
    - v1 (torch.Tensor): The second input tensor.
    - t (float): Interpolation parameter (0 <= t <= 1).
    - b (float): Threshold for the ratio between the norms of the input tensors.
    - t_sum (float): Interpolation parameter for the sum of the two tensors.

    Returns:
    - Tuple[torch.Tensor, float, float]: The interpolated tensor and the norms of the input tensors.
    """

    # Normalize tensors
    v0, norm_v0 = normalize_tensor(v0, device=device)
    v1, norm_v1 = normalize_tensor(v1, device=device)

    if norm_v1 < 0.0001:
        return v0, norm_v0, norm_v1

    if norm_v0 < 0.0001:
        # I really don't know what to do here
        logger.info(f"Warning: Small norm v0 ({norm_v0})")
        return v0, norm_v0, norm_v1

    # Compute FFT of the input tensors
    fft_v0 = fft_transform(v0, device=device)
    fft_v1 = fft_transform(v1, device=device)

    ratio = norm_v1 / (norm_v0 + 1e-10)

    # If we have really small values, they are probably noise, but might be important
    if ratio < b:
        logger.info(f"Small norm v1 ({norm_v1})")
        # We should just add the second tensor to the first one
        result_fft = fft_v0 + fft_v1 * t
    else:
        # Interpolate FFT components
        result_fft = interpolate_fft_components(
            fft_v0,
            fft_v1,
            t=t,
            t_sum=t_sum,
            cutoff_pct=cutoff_pct,
            cull_pct=cull_pct,
            device=device,
        )

    # Perform the inverse FFT to get back to the spatial domain
    merged_tensor = ifft_transform(result_fft, device=device)

    # check for nan/inf and print a report
    if torch.any(torch.isnan(merged_tensor)):
        merged_tensor = torch.where(
            torch.isnan(merged_tensor), torch.zeros_like(merged_tensor), merged_tensor
        )
        logger.info(f"Warning: NaN in ifft output: {torch.isnan(merged_tensor).sum()}")

    if torch.any(torch.isinf(merged_tensor)):
        logger.info(f"Warning: Inf in ifft output: {torch.isinf(merged_tensor).sum()}")
        raise ValueError("Inf in ifft output")

    del fft_v0, fft_v1, result_fft

    return merged_tensor, norm_v0, norm_v1


@merge_method(
    name="fourier_slerp",
    pretty_name="Fourier SLERP",
    reference_url="https://github.com/54rt1n/shardmerge",
)
def fourier_slerp(
    tensors: List[torch.Tensor],
    weight: List[float],
    target_norm_offset: Optional[float] = 1e-10,
):
    assert len(tensors) == 2, "Fourier SLERP requires exactly two tensors"

    a = tensors[0].to(torch.float32)
    a_weight = weight[0] / sum(weight)
    b = tensors[1].to(torch.float32)
    b_weight = weight[1] / sum(weight)
    t = 1 - a_weight

    norm_a = a.norm().item()
    norm_b = b.norm().item()
    target_norm = ((norm_a + norm_b) / 2) + target_norm_offset

    if abs(norm_a) < abs(norm_b):
        a, b = b, a
        norm_a, norm_b = norm_b, norm_a
        a_weight, b_weight = b_weight, a_weight

    cnorm_a = abs(norm_a / target_norm)
    cnorm_b = abs(norm_b / target_norm)
    n_ratio = cnorm_b / (cnorm_a + 1e-10)

    res = torch.zeros_like(a)

    if cnorm_a < 1e-6:
        res = (a * a_weight) + (b * b_weight)  # straight linear merging
    elif cnorm_b < 1e-6 or n_ratio < 0.1:
        norm_scale = target_norm / norm_a
        scaled_a = a * norm_scale
        weight_scale = b_weight / (a_weight + 1e-10)
        scaled_b = b * weight_scale * norm_scale
        res = (scaled_a * a_weight) + (scaled_b * b_weight)  # wonky linear merging
    else:
        res = merge_tensors_fft2_slerp(
            a, b, t=t, device=a.device, b=0.1, t_sum=1.0, cutoff_pct=0.0, cull_pct=0.0
        )
        res = res * target_norm

    del (
        a,
        b,
        a_weight,
        b_weight,
        cnorm_a,
        cnorm_b,
        n_ratio,
        norm_a,
        norm_b,
        target_norm,
        norm_scale,
        scaled_a,
        scaled_b,
        weight_scale,
    )

    return res.to(tensors[0].dtype)
