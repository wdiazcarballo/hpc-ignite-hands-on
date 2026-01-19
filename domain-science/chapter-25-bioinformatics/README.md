# บทที่ 25: ชีวสารสนเทศศาสตร์

Chapter 25: Bioinformatics

## วัตถุประสงค์การเรียนรู้

1. เข้าใจ DNA/Protein Sequences
2. ทำ Sequence Alignment
3. วิเคราะห์ Genomic Data
4. ใช้ BioPython

## โครงสร้างไฟล์

```
chapter-25-bioinformatics/
├── README.md
├── sequence_basics.py      # DNA/RNA/Protein basics
├── alignment.py            # Sequence alignment
├── blast_analysis.py       # BLAST search
├── phylogenetics.py        # Phylogenetic trees
└── sbatch/
    └── bioinfo_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-bioinfo python=3.9 biopython numpy matplotlib
mamba activate hpc-bioinfo

# Run examples
python sequence_basics.py
python alignment.py
```

## Key Concepts

- **DNA**: A, T, G, C nucleotides
- **RNA**: A, U, G, C (transcription)
- **Protein**: 20 amino acids (translation)
- **BLAST**: Basic Local Alignment Search Tool
