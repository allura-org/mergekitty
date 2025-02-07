# Copyright (C) 2025 Allura-org
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

from typing import List, Optional

import torch
from torch.fft import fft, ifft

from mergekitty.merge_methods.easy_define import merge_method
from mergekitty.merge_methods.nuslerp import nuslerp


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

    a = tensors[0]
    b = tensors[1]

    weight_normalizer = 1 / (weight[0] + weight[1])
    weight_a = weight[0] * weight_normalizer
    weight_b = weight[1] * weight_normalizer

    norm_a = torch.norm(a).item()
    norm_b = torch.norm(b).item()

    if abs(norm_a) < abs(norm_b):
        a, b = b, a
        norm_a, norm_b = norm_b, norm_a

    ratio = norm_b / (norm_a + 1e-10)
    target_norm = ((norm_a + norm_b) / 2) + target_norm_offset

    v0 = a / norm_a
    v1 = b / norm_b

    v0 = fft(v0.to(torch.float32))
    v1 = fft(v1.to(torch.float32))

    if ratio < 0.1:
        res = v0 * weight_a + v1 * weight_b
    else:
        res = torch.zeros_like(v0, device=v0.device, dtype=v0.dtype)

        real_v0 = v0.real.to(torch.float32)
        real_v1 = v1.real.to(torch.float32)

        res.real = nuslerp(weight_b / (weight_a + weight_b), real_v0, real_v1).to(
            torch.complex32
        )

        ires = torch.zeros_like(v0, device=v0.device, dtype=v0.dtype)
        i0 = fft(v0.imag).real.to(torch.float32)
        i1 = fft(v1.imag).real.to(torch.float32)

        ires.imag = ifft(nuslerp(weight_b / (weight_a + weight_b), i0, i1))

    res = ifft(res).real
    res = res * target_norm

    return res.to(a.dtype)
