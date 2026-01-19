#!/usr/bin/env python3
"""
Molecular Basics - พื้นฐาน Molecular Chemistry
Chapter 20: Computational Chemistry

การทำงานกับโครงสร้างโมเลกุลด้วย RDKit
"""

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not installed. Install with: mamba install rdkit")


def demonstrate_smiles():
    """สาธิต SMILES notation"""
    print("\n1. SMILES Notation:")
    print("   SMILES = Simplified Molecular Input Line Entry System")

    molecules = {
        'Water': 'O',
        'Ethanol': 'CCO',
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Paracetamol': 'CC(=O)NC1=CC=C(C=C1)O',
    }

    for name, smiles in molecules.items():
        print(f"   {name:12s}: {smiles}")

    if RDKIT_AVAILABLE:
        print("\n   Parsing SMILES with RDKit:")
        for name, smiles in molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                atoms = mol.GetNumAtoms()
                bonds = mol.GetNumBonds()
                print(f"   {name:12s}: {atoms} atoms, {bonds} bonds")


def calculate_properties():
    """คำนวณคุณสมบัติโมเลกุล"""
    print("\n2. Molecular Properties:")

    if not RDKIT_AVAILABLE:
        print("   [Requires RDKit]")
        return

    # Thai medicinal plant compounds
    compounds = {
        'Andrographolide': 'CC1=C(C2C(CC1)C3(CCCC(C3C2O)C)C)COC(=O)C=CC4=CC=CC=C4',
        'Curcumin': 'COC1=CC(=CC(=C1O)OC)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC',
        'Gingerol': 'CCCCC(CC(=O)CCC1=CC(=C(C=C1)O)OC)O',
    }

    print(f"\n   {'Compound':20s} {'MW':>8s} {'LogP':>8s} {'HBD':>5s} {'HBA':>5s}")
    print("   " + "-" * 48)

    for name, smiles in compounds.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            print(f"   {name:20s} {mw:8.2f} {logp:8.2f} {hbd:5d} {hba:5d}")


def lipinski_rule():
    """ตรวจสอบ Lipinski's Rule of Five"""
    print("\n3. Lipinski's Rule of Five:")
    print("   Drug-like compounds typically have:")
    print("   - MW <= 500 Da")
    print("   - LogP <= 5")
    print("   - H-bond donors <= 5")
    print("   - H-bond acceptors <= 10")

    if not RDKIT_AVAILABLE:
        print("\n   [Requires RDKit for analysis]")
        return

    # Test compounds
    test_compounds = [
        ('Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
        ('Caffeine', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
        ('Paracetamol', 'CC(=O)NC1=CC=C(C=C1)O'),
    ]

    print(f"\n   {'Compound':15s} {'MW':>6s} {'LogP':>6s} {'HBD':>4s} {'HBA':>4s} {'Pass':>6s}")
    print("   " + "-" * 45)

    for name, smiles in test_compounds:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # Check Lipinski
            passes = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
            status = "Yes" if passes else "No"

            print(f"   {name:15s} {mw:6.1f} {logp:6.2f} {hbd:4d} {hba:4d} {status:>6s}")


def molecular_fingerprints():
    """สาธิต Molecular Fingerprints"""
    print("\n4. Molecular Fingerprints:")
    print("   Binary vectors representing molecular features")

    if not RDKIT_AVAILABLE:
        print("   [Requires RDKit]")
        return

    mol1 = Chem.MolFromSmiles('CCO')  # Ethanol
    mol2 = Chem.MolFromSmiles('CCCO')  # Propanol
    mol3 = Chem.MolFromSmiles('c1ccccc1')  # Benzene

    # Generate Morgan fingerprints
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    fp3 = AllChem.GetMorganFingerprintAsBitVect(mol3, 2, nBits=1024)

    # Calculate Tanimoto similarity
    from rdkit import DataStructs
    sim12 = DataStructs.TanimotoSimilarity(fp1, fp2)
    sim13 = DataStructs.TanimotoSimilarity(fp1, fp3)
    sim23 = DataStructs.TanimotoSimilarity(fp2, fp3)

    print("\n   Tanimoto Similarity:")
    print(f"   Ethanol vs Propanol: {sim12:.3f} (similar)")
    print(f"   Ethanol vs Benzene:  {sim13:.3f} (different)")
    print(f"   Propanol vs Benzene: {sim23:.3f} (different)")


def main():
    print("=" * 60)
    print("   Molecular Basics")
    print("   Chapter 20: Computational Chemistry")
    print("=" * 60)

    demonstrate_smiles()
    calculate_properties()
    lipinski_rule()
    molecular_fingerprints()

    print("\n" + "=" * 60)
    print("   Molecular basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
