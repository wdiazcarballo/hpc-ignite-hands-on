# บทที่ 8: การเขียนโปรแกรม MPI ขั้นสูง

Chapter 8: Advanced MPI Programming

## วัตถุประสงค์การเรียนรู้

1. ใช้ MPI Collective Operations (Scatter, Gather, Allreduce)
2. ประยุกต์ Domain Decomposition
3. จัดการ Ghost Cells สำหรับ Stencil Operations
4. Optimize การสื่อสารด้วย Non-blocking MPI

## โครงสร้างไฟล์

```
chapter-08-mpi/
├── README.md
├── collective_ops.py        # Collective operations demo
├── domain_decomposition.py  # 1D/2D decomposition
├── heat_equation.py         # Heat diffusion simulation
├── ghost_cells.py           # Ghost cell exchange
└── sbatch/
    └── mpi_heat.sbatch
```

## MPI Collective Operations

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Broadcast: one-to-all
data = comm.bcast(data, root=0)

# Scatter: distribute array
local = comm.scatter(global_array, root=0)

# Gather: collect from all
global_array = comm.gather(local, root=0)

# Allreduce: reduce + broadcast
total = comm.allreduce(local_sum, op=MPI.SUM)
```
