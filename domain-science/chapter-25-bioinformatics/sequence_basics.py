#!/usr/bin/env python3
"""
Sequence Basics - พื้นฐาน DNA/Protein Sequences
Chapter 25: Bioinformatics

การทำงานกับลำดับ DNA, RNA และโปรตีน
"""

import numpy as np
from collections import Counter

try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction, molecular_weight
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("BioPython not installed. Install with: pip install biopython")


# Codon table (Standard genetic code)
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def demonstrate_dna_basics():
    """สาธิตพื้นฐาน DNA"""
    print("\n1. DNA Basics:")

    # Example: Part of Thai jasmine rice Wx gene
    dna_seq = "ATGGCGCCCAAGCTCGTGCTCCTCTCCCTGCTCCTCGCCGCC"

    print(f"   DNA sequence (5' to 3'):")
    print(f"   {dna_seq}")
    print(f"   Length: {len(dna_seq)} bp")

    # Nucleotide composition
    counts = Counter(dna_seq)
    print(f"\n   Composition:")
    for base in 'ATGC':
        pct = counts[base] / len(dna_seq) * 100
        print(f"   {base}: {counts[base]} ({pct:.1f}%)")

    # GC content
    gc_content = (counts['G'] + counts['C']) / len(dna_seq) * 100
    print(f"\n   GC content: {gc_content:.1f}%")

    return dna_seq


def complement_and_reverse(dna_seq: str):
    """หา Complement และ Reverse Complement"""
    print("\n2. Complementary Strands:")

    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

    complement = ''.join(complement_map[base] for base in dna_seq)
    reverse_complement = complement[::-1]

    print(f"   5' {dna_seq[:30]}... 3'  (Original)")
    print(f"   3' {complement[:30]}... 5'  (Complement)")
    print(f"   5' {reverse_complement[:30]}... 3'  (Reverse Complement)")

    return complement, reverse_complement


def transcribe_to_rna(dna_seq: str):
    """Transcription: DNA → RNA"""
    print("\n3. Transcription (DNA → RNA):")

    # Replace T with U
    rna_seq = dna_seq.replace('T', 'U')

    print(f"   DNA: {dna_seq[:30]}...")
    print(f"   RNA: {rna_seq[:30]}...")
    print(f"   (T replaced with U)")

    return rna_seq


def translate_to_protein(dna_seq: str):
    """Translation: DNA → Protein"""
    print("\n4. Translation (DNA → Protein):")

    protein = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3]
        amino_acid = CODON_TABLE.get(codon, 'X')
        if amino_acid == '*':  # Stop codon
            break
        protein.append(amino_acid)

    protein_seq = ''.join(protein)

    print(f"   DNA:     {dna_seq[:30]}...")
    print(f"   Codons:  {' '.join(dna_seq[i:i+3] for i in range(0, min(30, len(dna_seq)), 3))}...")
    print(f"   Protein: {protein_seq[:10]}...")
    print(f"   Length: {len(protein_seq)} amino acids")

    return protein_seq


def find_orfs(dna_seq: str, min_length: int = 30):
    """Find Open Reading Frames (ORFs)"""
    print("\n5. Finding Open Reading Frames (ORFs):")

    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]

    orfs = []

    # Search all three reading frames
    for frame in range(3):
        i = frame
        while i < len(dna_seq) - 2:
            codon = dna_seq[i:i+3]
            if codon == start_codon:
                # Found start, look for stop
                for j in range(i + 3, len(dna_seq) - 2, 3):
                    stop_codon = dna_seq[j:j+3]
                    if stop_codon in stop_codons:
                        orf_length = j - i + 3
                        if orf_length >= min_length:
                            orfs.append({
                                'frame': frame + 1,
                                'start': i + 1,
                                'end': j + 3,
                                'length': orf_length
                            })
                        break
            i += 3

    print(f"   Minimum ORF length: {min_length} bp")
    print(f"   ORFs found: {len(orfs)}")

    for orf in orfs[:5]:  # Show first 5
        print(f"   Frame {orf['frame']}: {orf['start']}-{orf['end']} ({orf['length']} bp)")

    return orfs


def calculate_protein_properties(protein_seq: str):
    """คำนวณคุณสมบัติโปรตีน"""
    print("\n6. Protein Properties:")

    # Amino acid properties
    hydrophobic = set('AILMFWYV')
    polar = set('STNQ')
    charged = set('DEKRH')

    counts = Counter(protein_seq)
    total = len(protein_seq)

    hydro_pct = sum(counts[aa] for aa in hydrophobic) / total * 100
    polar_pct = sum(counts[aa] for aa in polar) / total * 100
    charged_pct = sum(counts[aa] for aa in charged) / total * 100

    print(f"   Length: {total} amino acids")
    print(f"   Hydrophobic: {hydro_pct:.1f}%")
    print(f"   Polar: {polar_pct:.1f}%")
    print(f"   Charged: {charged_pct:.1f}%")

    # Molecular weight estimate (average ~110 Da per amino acid)
    mw_estimate = total * 110
    print(f"   Est. molecular weight: {mw_estimate:,} Da")


def demonstrate_biopython():
    """สาธิตการใช้ BioPython"""
    print("\n7. BioPython Features:")

    if not BIOPYTHON_AVAILABLE:
        print("   [Requires BioPython]")
        return

    seq = Seq("ATGGCGCCCAAGCTCGTGCTCCTCTCCCTGCTCCTCGCCGCC")

    print(f"   Sequence: {seq[:30]}...")
    print(f"   Length: {len(seq)}")
    print(f"   GC content: {gc_fraction(seq)*100:.1f}%")
    print(f"   Complement: {seq.complement()[:30]}...")
    print(f"   Reverse complement: {seq.reverse_complement()[:30]}...")
    print(f"   Transcribed: {seq.transcribe()[:30]}...")
    print(f"   Translated: {seq.translate()[:10]}...")


def main():
    print("=" * 60)
    print("   Sequence Basics - DNA/RNA/Protein")
    print("   Chapter 25: Bioinformatics")
    print("=" * 60)

    # DNA basics
    dna_seq = demonstrate_dna_basics()

    # Complement
    complement_and_reverse(dna_seq)

    # Transcription
    transcribe_to_rna(dna_seq)

    # Translation
    protein_seq = translate_to_protein(dna_seq)

    # Find ORFs
    find_orfs(dna_seq)

    # Protein properties
    calculate_protein_properties(protein_seq)

    # BioPython
    demonstrate_biopython()

    print("\n" + "=" * 60)
    print("   Sequence basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
