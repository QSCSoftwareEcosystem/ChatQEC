"""Circuit-level memory-experiment generators from decoder-bench, copied verbatim.

Source: https://github.com/satvikmaurya/decoder-bench, commit
9733a55a3a0e37c618294bd2ebbe0a8d4c334943, decoder_bench/generator.py (MIT, see LICENSE in this
directory). Only the three circuit-noise generators the chatQEC evaluation scores against are
kept; the imports are rewritten to this package. Do not edit the function bodies: their output
is checked against circuits made by the original code.
"""
# ruff: noqa
# mypy: ignore-errors
from typing import Tuple

import stim

from .build_circuit import build_circuit
from .codes_q import create_bivariate_bicycle_codes
from .color_code_gen import ColorCode
from .noise import NoiseModel

def gen_surface_circuit(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    circuit = stim.Circuit.generated(f'surface_code:rotated_memory_{basis}',
                                     rounds=d,
                                     distance=d)
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit

def gen_color_circuit(input:Tuple[int, float]) -> stim.Circuit:
    d, p = input
    circuit = ColorCode(d=d, rounds=d, p_bitflip=p).circuit
    circuit = circuit.without_noise()
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit

def gen_qldpc_circuit(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    if d == 6:
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
    elif d == 10:
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
    elif d == 12:
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
    else:
        raise ValueError('Invalid distance. Only 6, 10, and 12 are supported.')
    circuit = build_circuit(code, A_list, B_list, p=0,
                            num_repeat=d, use_both=False, 
                            z_basis=(basis == 'z'))
    circuit = circuit.without_noise()
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit
