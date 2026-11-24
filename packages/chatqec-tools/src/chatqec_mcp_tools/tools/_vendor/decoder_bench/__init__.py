"""Vendored subset of decoder-bench (MIT): circuit generators and the SI1000 noise model.

Source: https://github.com/satvikmaurya/decoder-bench at commit
9733a55a3a0e37c618294bd2ebbe0a8d4c334943 (paper: IEEE 11242095). Copyright (c) 2025 Satvik
Maurya; the MIT license text is in LICENSE next to this file.

Files are copied verbatim except for import lines (marked "vendored") and igraph, which is
GPL-licensed and replaced by the minimal shim in _graph.py. tests/test_decoder_bench_vendor.py
checks that the generated circuits are byte-identical to the original code's.
"""
