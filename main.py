import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
from termcolor import colored
import ctypes
import os
from scipy.sparse.csgraph import reverse_cuthill_mckee

def differences(nx,nz):
    dx=1/(nx-1)
    dz=1/(nz-1)
    nelem=nz*nx

    # Initialize sparse matrix components
    diagonals = []
    diagonal_positions = []

    # Main diagonal (interior points)
    main_diag = 2 / dx**2 + 2 / dz**2
    diagonals.append(np.full(nelem, main_diag))
    diagonal_positions.append(0)

    # Off-diagonals for x-direction (left and right neighbors)
    left_diag = -1 / dx**2
    right_diag = -1 / dx**2
    left_positions = np.full(nelem - 1, left_diag)
    right_positions = np.full(nelem - 1, right_diag)

    diagonals.append(left_positions)
    diagonal_positions.append(-1)
    diagonals.append(right_positions)
    diagonal_positions.append(1)

    # Off-diagonals for z-direction (up and down neighbors)
    up_diag = -1 / dz**2
    down_diag = -1 / dz**2

    diagonals.append(np.full(nelem - nx, up_diag))
    diagonal_positions.append(-nx)
    diagonals.append(np.full(nelem - nx, down_diag))
    diagonal_positions.append(nx)

    # Assemble sparse matrix A
    A = sp.diags(diagonals, diagonal_positions, shape=(nelem, nelem), format="lil")

    # Update matrix A to reflect boundary conditions
    A[0:nx, :] = 0
    A[:, 0:nx] = 0
    A[0:nx, 0:nx] = sp.eye(nx)          # Bottom boundary
    A[-nx:, :] = 0
    A[:, -nx:] = 0
    A[-nx:, -nx:] = sp.eye(nx)          # Top boundary
    A[::nx, :] = 0
    A[:, ::nx] = 0
    A[::nx, ::nx] = sp.eye(nz)          # Left boundary
    A[nx-1::nx, :] = 0
    A[:, nx-1::nx] = 0
    A[nx-1::nx, nx-1::nx] = sp.eye(nz)  # Right boundary

    return A

def dense_to_coo(L):
    n=L.shape[0]
    L_nnz=0
    L_data=np.zeros(20*n)
    L_r=np.zeros(20*n, dtype=np.int32)
    L_c=np.zeros(20*n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if (np.abs(L[i][j])>1e-10):
                L_data[L_nnz]=L[i][j]
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz


# seed = 42
# rng = np.random.default_rng(seed)
# np.random.seed(seed)
# n=20000
# random_matrix = sp.random(n, n, density=5/n, format='csr', data_rvs=lambda n: np.random.choice([0, 1], size=n), random_state=rng)
# random_matrix.setdiag(7)
# A=random_matrix.toarray()
# A+=A.T
# A_sparse=csr_matrix(A)

n=300
A=differences(n,n)
A_sparse=A.tocsr()
n=A_sparse.shape[0]

print(colored("-------------------------------------------------------------", "cyan"))
print(colored("Matrix A:", "cyan"))
print(colored("n = {:d}".format(n), "cyan"))
print(colored("nnz = {:d}".format(A_sparse.nnz), "cyan"))
print(colored("-------------------------------------------------------------", "cyan"))

start=time.time()
order=reverse_cuthill_mckee(A_sparse, symmetric_mode=True)
end=time.time()
print(colored("Time rcmk : {:f}".format(end-start), "red"))

A_sparse = A_sparse.tolil()
# A_reordered = A_sparse[order, :][:, order]
A_reordered = A_sparse
# A_reordered = A_reordered.toarray()
A_reordered = A_reordered.tocoo()
A_data = A_reordered.data
A_r = A_reordered.row
A_c = A_reordered.col
A_nnz = A_reordered.nnz

if (False):
    plt.spy(A_reordered, markersize=1)
    plt.show()

# A_data,A_r,A_c,A_nnz=dense_to_coo(A_reordered)
b=np.ones(n)
x=np.zeros(n)

# print(colored("A_nnz : {:d}".format(A_nnz), "red"))

os.system("make clean")
os.system("make")

lib = ctypes.CDLL('./main.so')
c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
lib.run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_int_p, c_int_p, c_double_p, c_double_p, c_double_p]

nnz_p = ctypes.c_int(A_nnz)
m_p = ctypes.c_int(n)
n_p = ctypes.c_int(n)
row_p = A_r.ctypes.data_as(c_int_p)
col_p = A_c.ctypes.data_as(c_int_p)
data_p = A_data.ctypes.data_as(c_double_p)
B_p = b.ctypes.data_as(c_double_p)
x_p = x.ctypes.data_as(c_double_p)

start = time.time()
lib.run(nnz_p, m_p, n_p, row_p, col_p, data_p, B_p, x_p)
end = time.time()
x = np.ctypeslib.as_array(x_p, (n,))
print(colored("Time solver : {:f}".format(end-start), "red"))

start = time.time()
x_np = sp.linalg.spsolve(A_reordered.tocsr(), b)
end = time.time()
print(colored("Time scipy : {:f}".format(end-start), "red"))


# Compare to scipy
for i in range(n):
    if (abs(x[i]-x_np[i])>1e-6):
        print(colored("Error", "red"))
        exit()

print(colored("Exit", "green"))
