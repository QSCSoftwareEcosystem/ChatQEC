# Surface codes hit a milestone

Posted by alice on 2026-01-15.

## What changed?

Google's Willow chip showed the surface code below threshold for the first time. This achievement represents a fundamental breakthrough in quantum error correction, marking the first experimental demonstration of exponential suppression of logical error rates with increasing physical qubits. The team created a two-dimensional array of superconducting qubits and implemented a surface code protocol that corrects errors faster than they accumulate.

## Why it matters

This is the demonstration the field has been waiting for since the 1990s. Surface codes are a topological approach to quantum error correction, which theoretically requires only local interactions between physical qubits and offers strong noise resilience. Previous experiments had shown below-threshold behavior at small scales, but with significant experimental overhead. Willow's demonstration at meaningful scale validates decades of theoretical work.

## Technical approach

The surface code implementation uses a two-dimensional grid of superconducting qubits. Each qubit is coupled to four neighbors through adjustable couplings. The code measures stabilizer operators non-destructively to extract syndrome information about errors without collapsing quantum information. Syndrome extraction uses a repetition code pattern with measurement feedback.

## Caveats

Single-qubit logic, not yet a full computer. Still a huge step forward. The demonstration involves specific error models favorable to surface codes. Scaling to fault-tolerant universal quantum computation requires additional engineering challenges including longer coherence times and higher two-qubit gate fidelities. The current chip operates at dilution refrigerator temperatures and requires careful noise management.
