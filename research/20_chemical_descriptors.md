# Research: Chemical Descriptor & Molecular Fingerprint Atoms

## Goal

Find best-in-class, pure-function implementations for computing molecular
descriptors and fingerprints from SMILES strings. Target repo: `sciona-atoms-bio`.

## CDG stages this research covers (1 stage)

- `neurips_open_polymer_1st/chemical_descriptor_extraction`: Process SMILES
  chemical strings to structurally generate RDKit molecular descriptors and
  Morgan fingerprints (rdkit)

## What to research

### 1. SMILES to molecular graph
- `smiles_to_mol(smiles: str) -> MolObject` — RDKit Mol object construction
- Pure function: validate SMILES, construct graph
- Source: RDKit (BSD-3-Clause)

### 2. Molecular descriptors
- `compute_descriptors(mol: MolObject, descriptor_list: list[str]) -> NDArray`
- Standard descriptors: molecular weight, logP, TPSA, number of H-bond donors/acceptors,
  number of rotatable bonds, aromatic ring count
- Source: RDKit Descriptors module (BSD-3-Clause)

### 3. Morgan fingerprints (circular fingerprints / ECFP)
- `morgan_fingerprint(mol: MolObject, radius: int, n_bits: int) -> NDArray`
- Extended-Connectivity Fingerprint (ECFP4 when radius=2)
- Source: RDKit AllChem.GetMorganFingerprintAsBitVect (BSD-3-Clause)

### 4. MACCS keys
- `maccs_keys(mol: MolObject) -> NDArray`
- 166-bit structural key fingerprint
- Source: RDKit MACCSkeys module

## Research questions

1. Can we wrap RDKit functions as pure atoms? (RDKit is BSD-3, acceptable)
2. What is the standard descriptor set for competition use?
   (Top-20 most discriminative descriptors from QSAR literature)
3. Should SMILES parsing be a separate atom from descriptor computation?
   (Yes — parse once, compute multiple descriptor types)
4. What contracts are natural? (valid SMILES → non-null mol,
   fingerprint length == n_bits, descriptors are finite)

## Output format

Concept types: `data_extraction` for SMILES parsing, `analysis` for descriptor
computation.

For each candidate atom, provide:
```
Name: morgan_fingerprint
Description: Compute a Morgan (ECFP) circular fingerprint bit vector from an
  RDKit molecule object.
Source: URL to the best reference implementation or paper
License: BSD-3-Clause (RDKit)
Concept type: analysis
Signature: (mol: Mol, radius: int, n_bits: int) -> NDArray
Pure function boundary: molecule object and explicit parameters in, fixed-length
  bit vector out; no file I/O, database queries, or global state.
Contracts:
  - require: mol is not None (valid molecule)
  - require: radius >= 0
  - require: n_bits > 0
  - ensure: result.shape == (n_bits,)
  - ensure: result values are 0 or 1
Witness: ethanol SMILES "CCO" with radius=2, n_bits=1024; verify known bit
  positions for hydroxyl substructure.
Dependencies: rdkit (BSD-3-Clause)
CDG stages covered: neurips_open_polymer_1st/chemical_descriptor_extraction
```
