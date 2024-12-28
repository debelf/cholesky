from multiprocessing import Process
import time
from termcolor import colored
import numpy as np
from scipy.linalg import cholesky
import multiprocessing as mp
import scipy.sparse as sp

def calculate_square():
    for i in range(20000):
        for j in range(i):
            square = j ** 2

def calculate_cube():
    for i in range(20000):
        for j in range(i):
            cube = j ** 3

def coo_to_dense(L_data,L_r,L_c,n):
    L=np.zeros((n,n))
    for i in range(len(L_data)):
        L[int(L_r[i])][int(L_c[i])]=L_data[i]
    return L

def cholesky_block(A, block_size):
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(0, n, block_size):
        # Taille du bloc courant
        end_i = min(i + block_size, n)
        
        # Bloc diagonal
        A_block = A[i:end_i, i:end_i]
        L[i:end_i, i:end_i] = cholesky(A_block, lower=True)
        
        # Blocs hors diagonale
        for j in range(end_i, n, block_size):
            end_j = min(j + block_size, n)
            A_sub = A[j:end_j, i:end_i]
            L[j:end_j, i:end_i] = np.dot(A_sub, np.linalg.inv(L[i:end_i, i:end_i].T))
        
        print("ici",np.linalg.inv(L[i:end_i, i:end_i].T))
        # Mise à jour du reste
        for j in range(end_i, n, block_size):
            end_j = min(j + block_size, n)
            A[j:end_j, j:end_j] -= np.dot(L[j:end_j, i:end_i], L[j:end_j, i:end_i].T)    
    return L

def cholesky_crout(A):
    # column by column
    n=A.shape[0]
    L=np.zeros((n,n))
    for j in range(n):
        sum=0
        for k in range(j):
            sum+=L[j][k]*L[j][k]
        L[j][j]=np.sqrt(A[j][j]-sum)
        for i in range(j+1,n):
            sum=0
            for k in range(j):
                sum+=L[i][k]*L[j][k]
            L[i][j]=(A[i][j]-sum)/L[j][j]
    return L

def cholesky_crout_coo(n, A):
    L_nnz=0
    L_data=np.zeros(n*n)
    L_r=np.zeros(n*n, dtype=int)
    L_c=np.zeros(n*n, dtype=int)
    for j in range(n):
        sum=0
        for index in range(L_nnz):
            if (L_r[index]==j and L_c[index]<j):
                sum+=L_data[index]*L_data[index]
        calc=np.sqrt(A[j][j]-sum)
        if (calc>0):
            L_data[L_nnz]=calc
            L_r[L_nnz]=j
            L_c[L_nnz]=j
            L_nnz+=1

        for i in range(j+1,n):
            sum=0
            for index2 in range(L_nnz):
                if (L_r[index2]==j):
                    for index in range(index2,L_nnz):
                        if (L_r[index]==i):
                            if (L_c[index2]==L_c[index]):
                                sum+=L_data[index]*L_data[index2]
            L_den=0
            for index2 in range(L_nnz):
                if (L_r[index2]==j and L_c[index2]==j):
                    L_den=L_data[index2]
            calc=(A[i][j]-sum)/L_den
            if (np.abs(calc)>0):
                L_data[L_nnz]=calc
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_crout4_coo(n, A):
    L_nnz=0
    L_data=np.zeros(n*n)
    L_r=np.zeros(n*n)
    L_c=np.zeros(n*n)

    for j in range(n):
        sum=0
        current_j=0
        for index2 in range(L_nnz):
            if (L_r[index2]==j and L_c[index2]<j):
                if (current_j==0):
                    current_j=index2 
                sum+=L_data[index2]*L_data[index2]
        calc=np.sqrt(A[j][j]-sum)
        if (calc>0):
            L_data[L_nnz]=calc
            L_r[L_nnz]=j
            L_c[L_nnz]=j
            L_nnz+=1
        L_den=0
        for index2 in range(current_j,L_nnz):
            if (L_r[index2]==j and L_c[index2]==j):
                L_den=L_data[index2]
                break
        for i in range(j+1,n):
            sum=0
            for index2 in range(current_j,L_nnz):
                if (L_r[index2]==j):
                    for index in range(index2,L_nnz):
                        if (L_r[index]>=i):
                            if (L_r[index]==i):
                                if (L_c[index2]<=L_c[index]):
                                    if (L_c[index2]==L_c[index]):
                                        sum+=L_data[index]*L_data[index2]
                                    else:
                                        break
                            else:
                                break 
            calc=(A[i][j]-sum)/L_den
            if (np.abs(calc)>0):
                L_data[L_nnz]=calc
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_crout_parallel(A):
    n = A.shape[0]
    L = mp.Array('d', n * n)  # Shared memory array for L (1D flat array)
    L_np = np.frombuffer(L.get_obj()).reshape((n, n))  # Correctly access shared memory as NumPy array

    def compute_Li(j, i_start, i_end):
        for i in range(i_start, i_end):
            sum_k = sum(L_np[i][k] * L_np[j][k] for k in range(j))
            L_np[i][j] = (A[i][j] - sum_k) / L_np[j][j]

    for j in range(n):
        # Compute L[j][j]
        sum_k = sum(L_np[j][k] * L_np[j][k] for k in range(j))
        L_np[j][j] = np.sqrt(A[j][j] - sum_k)

        # Parallelize computation of L[i][j] for i > j
        if j + 1 < n:
            processes = []
            num_processes = mp.cpu_count()
            chunk_size = (n - (j + 1)) // num_processes + 1
            for p in range(num_processes):
                i_start = j + 1 + p * chunk_size
                i_end = min(j + 1 + (p + 1) * chunk_size, n)
                if i_start < i_end:
                    proc = mp.Process(target=compute_Li, args=(j, i_start, i_end))
                    processes.append(proc)
                    proc.start()

            for proc in processes:
                proc.join()

    return np.array(L_np)



def cholesky_crout_coo_parallel2(n, A):
    L_data = mp.Array('d', n * n)  # Shared memory for L_data
    L_r = mp.Array('i', n * n)  # Shared memory for row indices
    L_c = mp.Array('i', n * n)  # Shared memory for column indices
    L_nnz = mp.Value('i', 0)  # Shared counter for non-zero elements

    def compute_Li(j, i_start, i_end, L_nnz_old, begin_i, end_i, L_den):
        """Compute L[i][j] for i_start <= i < i_end."""
        local_nnz = []
        
        # print(begin_i)
        
        for i in range(i_start, i_end):
            sum_k = 0
            # for index2 in range(current_j,end_i[j]):
            #     if L_r[index2] == j:
            #         k=L_c[index2]
            #         for index in range(begin_i[i],end_i[i]):
            #             if (L_c[index]>k):
            #                 break
            #             if L_r[index] == i and k == L_c[index]:
            #                 sum_k += L_data[index] * L_data[index2]   
            #             # if L_c[index2] > L_c[index]:
            #             #     break
            for index in range(begin_i[i],end_i[i]):
                if L_r[index] == i:
                    k=L_c[index]
                    for index2 in range(begin_i[j],end_i[j]):
                        if L_r[index2] == j and L_c[index2] == k:
                            sum_k += L_data[index] * L_data[index2]   

            calc = (A[i][j] - sum_k) / L_den
            if np.abs(calc) > 0:
                local_nnz.append((i, j, calc))

        # Append results back to shared memory
        with L_nnz.get_lock():
            idx = L_nnz.value
            for i, j, calc in local_nnz:
                L_data[idx] = calc
                L_r[idx] = i
                L_c[idx] = j
                L_nnz.value += 1
                idx += 1

    for j in range(n):
        # Compute L[j][j]
        sum_k = 0
        for index in range(L_nnz.value):
            if L_r[index] == j and L_c[index] < j:
                sum_k += L_data[index] * L_data[index]

        calc = np.sqrt(A[j][j] - sum_k)
        if calc > 0:
            # with L_nnz.get_lock():
                idx = L_nnz.value
                L_data[idx] = calc
                L_r[idx] = j
                L_c[idx] = j
                L_nnz.value += 1

        # Parallelize L[i][j] computation for i > j
        if j + 1 < n:
            processes = []
            num_processes = 2
            chunk_size = (n - (j + 1)) // num_processes + 1
            L_nnz_old = L_nnz.value
            for p in range(num_processes):
                i_start = j + 1 + p * chunk_size
                i_end = min(j + 1 + (p + 1) * chunk_size, n)
                L_den = 0
                begin_i = np.zeros(n,dtype=int)+L_nnz_old
                end_i = np.zeros(n,dtype=int)
                for index2 in range(L_nnz_old):
                    val=L_r[index2]
                    begin_i[val] = min(begin_i[val], index2)
                    end_i[val] = max(end_i[val], index2+1)
                    if L_r[index2] == j:
                        if L_c[index2] == j:
                            L_den = L_data[index2]
                if i_start < i_end:
                    proc = mp.Process(target=compute_Li, args=(j, i_start, i_end, L_nnz_old, begin_i, end_i, L_den))
                    processes.append(proc)
                    proc.start()

            for proc in processes:
                proc.join()

    return np.array(L_data[:L_nnz.value]), np.array(L_r[:L_nnz.value], dtype=int), np.array(L_c[:L_nnz.value], dtype=int), L_nnz.value

def cholesky_crout_coo_parallel3(n, A):
    L_data = mp.Array('d', n * n)  # Shared memory for L_data
    L_r = mp.Array('i', n * n)  # Shared memory for row indices
    L_c = mp.Array('i', n * n)  # Shared memory for column indices
    L_nnz = mp.Value('i', 0)  # Shared counter for non-zero elements

    def compute_Li(j, i_start, i_end, L_nnz_old):
        """Compute L[i][j] for i_start <= i < i_end."""
        local_nnz = []
        L_den = 0
        current_j = L_nnz_old
        begin_i = np.zeros(n,dtype=int)+L_nnz_old
        end_i = np.zeros(n,dtype=int)
        for index2 in range(L_nnz_old):
            val=L_r[index2]
            begin_i[val] = min(begin_i[val], index2)
            end_i[val] = max(end_i[val], index2+1)
            if L_r[index2] == j:
                current_j=min(current_j, index2)
                if L_c[index2] == j:
                    L_den = L_data[index2]
        
        # print(begin_i)
        
        for i in range(i_start, i_end):
            sum_k = 0
            # for index2 in range(current_j,end_i[j]):
            #     if L_r[index2] == j:
            #         k=L_c[index2]
            #         for index in range(begin_i[i],end_i[i]):
            #             if (L_c[index]>k):
            #                 break
            #             if L_r[index] == i and k == L_c[index]:
            #                 sum_k += L_data[index] * L_data[index2]   
            #             # if L_c[index2] > L_c[index]:
            #             #     break
            for index in range(L_nnz_old):
                if L_r[index] == i:
                    k=L_c[index]
                    for index2 in range(L_nnz_old):
                        # if (L_c[index2]>k):
                        #     break
                        if L_r[index2] == j and L_c[index2] == k:
                            sum_k += L_data[index] * L_data[index2]   

            calc = (A[i][j] - sum_k) / L_den
            if np.abs(calc) > 0:
                local_nnz.append((i, j, calc))

        # Append results back to shared memory
        with L_nnz.get_lock():
            for i, j, calc in local_nnz:
                idx = L_nnz.value
                L_data[idx] = calc
                L_r[idx] = i
                L_c[idx] = j
                L_nnz.value += 1

    for j in range(n):
        # Compute L[j][j]
        sum_k = 0
        for index in range(L_nnz.value):
            if L_r[index] == j and L_c[index] < j:
                sum_k += L_data[index] * L_data[index]

        calc = np.sqrt(A[j][j] - sum_k)
        if calc > 0:
            # with L_nnz.get_lock():
                idx = L_nnz.value
                L_data[idx] = calc
                L_r[idx] = j
                L_c[idx] = j
                L_nnz.value += 1

        # Parallelize L[i][j] computation for i > j
        if j + 1 < n:
            processes = []
            num_processes = 10
            chunk_size = (n - (j + 1)) // num_processes + 1
            L_nnz_old = L_nnz.value
            for p in range(num_processes):
                i_start = j + 1 + p * chunk_size
                i_end = min(j + 1 + (p + 1) * chunk_size, n)
                if i_start < i_end:
                    proc = mp.Process(target=compute_Li, args=(j, i_start, i_end, L_nnz_old))
                    processes.append(proc)
                    proc.start()

            for proc in processes:
                proc.join()

    return np.array(L_data[:L_nnz.value]), np.array(L_r[:L_nnz.value], dtype=int), np.array(L_c[:L_nnz.value], dtype=int), L_nnz.value



def cholesky_crout_coo_parallel(n, A):
    L_data = mp.Array('d', n * n)  # Shared memory for L_data
    L_r = mp.Array('i', n * n)  # Shared memory for row indices
    L_c = mp.Array('i', n * n)  # Shared memory for column indices
    L_nnz = mp.Value('i', 0)  # Shared counter for non-zero elements
    queue = mp.Queue()  # Queue for storing local results

    def compute_Li(j, i_start, i_end, L_nnz_old, begin_i, end_i, L_den, queue):
        """Compute L[i][j] for i_start <= i < i_end."""
        local_results = []
        # L_den = 0
        # current_j = L_nnz_old
        # begin_i = np.zeros(n,dtype=int)+L_nnz_old
        # end_i = np.zeros(n,dtype=int)
        # for index2 in range(L_nnz_old):
        #     val=L_r[index2]
        #     begin_i[val] = min(begin_i[val], index2)
        #     end_i[val] = max(end_i[val], index2+1)
        #     if L_r[index2] == j:
        #         current_j=min(current_j, index2)
        #         if L_c[index2] == j:
        #             L_den = L_data[index2]
        # print(begin_i)
        
        for i in range(i_start, i_end):
            sum_k = 0
            # for index2 in range(current_j,end_i[j]):
            #     if L_r[index2] == j:
            #         k=L_c[index2]
            #         for index in range(begin_i[i],end_i[i]):
            #             if (L_c[index]>k):
            #                 break
            #             if L_r[index] == i and k == L_c[index]:
            #                 sum_k += L_data[index] * L_data[index2]   
            #             # if L_c[index2] > L_c[index]:
            #             #     break
            for index in range(begin_i[i],end_i[i]):
                if (L_r[index] == i):
                    k=L_c[index]
                    for index2 in range(begin_i[j],end_i[j]):
                        if (L_c[index2]>k):
                            break
                        if (L_r[index2] == j and L_c[index2] == k):
                            sum_k += L_data[index] * L_data[index2]   

            calc = (A[i][j] - sum_k) / L_den
            if np.abs(calc) > 0:
                local_results.append((i, j, calc))

        # Append results back to shared memory
        # with L_nnz.get_lock():
        #     idx = L_nnz.value
        #     for i, j, calc in local_nnz:
        #         L_data[idx] = calc
        #         L_r[idx] = i
        #         L_c[idx] = j
        #         L_nnz.value += 1
        #         idx += 1
        queue.put(local_results)


    for j in range(n):
        # Compute L[j][j]
        sum_k = 0
        for index in range(L_nnz.value):
            if L_r[index] == j and L_c[index] < j:
                sum_k += L_data[index] * L_data[index]

        calc = np.sqrt(A[j][j] - sum_k)
        if calc > 0:
            # with L_nnz.get_lock():
                idx = L_nnz.value
                L_data[idx] = calc
                L_r[idx] = j
                L_c[idx] = j
                L_nnz.value += 1
        
        L_den = 0
        L_nnz_old = L_nnz.value
        begin_i = np.zeros(n,dtype=int) + L_nnz_old
        end_i = np.zeros(n,dtype=int)
        for index2 in range(L_nnz_old):
            val=L_r[index2]
            begin_i[val] = min(begin_i[val], index2)
            end_i[val] = max(end_i[val], index2+1)
            if L_r[index2] == j:
                if L_c[index2] == j:
                    L_den = L_data[index2]

        # Parallelize L[i][j] computation for i > j
        if j + 1 < n:
            processes = []
            num_processes = 4
            chunk_size = (n - (j + 1)) // num_processes + 1
            for p in range(num_processes):
                i_start = j + 1 + p * chunk_size
                i_end = min(j + 1 + (p + 1) * chunk_size, n)
                if i_start < i_end:
                    proc = mp.Process(target=compute_Li, args=(j, i_start, i_end, L_nnz_old, begin_i, end_i, L_den, queue))
                    processes.append(proc)
                    proc.start()

            for proc in processes:
                proc.join()
            
            while not queue.empty():
                results = queue.get()
                # with L_nnz.get_lock():
                for i, j, calc in results:
                    idx = L_nnz.value
                    L_data[idx] = calc
                    L_r[idx] = i
                    L_c[idx] = j
                    L_nnz.value += 1

    return np.array(L_data[:L_nnz.value]), np.array(L_r[:L_nnz.value], dtype=int), np.array(L_c[:L_nnz.value], dtype=int), L_nnz.value


# A = np.array([[4, 2], [2, 3]],dtype=float)
# L_expected = np.linalg.cholesky(A)
# L_block = cholesky_block(A, block_size=1)
# print("Expected:\n", L_expected)
# print("Block Result:\n", L_block)

# n=50
# A = np.random.rand(n, n)
# A += A.T
# A += n * np.eye(n)  

seed = 42
rng = np.random.default_rng(seed)
np.random.seed(seed)
n=1000
random_matrix = sp.random(n, n, density=1.05/n, format='csr', data_rvs=lambda n: np.random.choice([0, 1], size=n), random_state=rng)
random_matrix.setdiag(7)
A=random_matrix.toarray()
A+=A.T

# start=time.time()
# L_data, L_r, L_c, L_nnz = cholesky_crout_coo(n, A)
# end=time.time()
# print("Time normal", end-start)

print(colored("-------------------------------------------------------------", "cyan"))

start=time.time()
L_par = cholesky_crout_parallel(A)
end=time.time()
print("Time parallel", end-start)


start=time.time()
Ln_data,Ln_r,Ln_c,Ln_nnz = cholesky_crout_coo_parallel(n,A)
end=time.time()
print("Time parallel", end-start)


# start=time.time()
# import line_profiler
# lp = line_profiler.LineProfiler()
# lp_wrapper = lp(cholesky_crout_coo)
# # lp.add_function(compute_Li)
# Ln_data,Ln_r,Ln_c,Ln_nnz=lp_wrapper(n,A)
# lp.print_stats()
# end=time.time()
# print(colored("Time : {:f}".format(end-start), "red"))

# L_nor = coo_to_dense(L_data,L_r,L_c,n)
L_par2 = coo_to_dense(Ln_data,Ln_r,Ln_c,n) 
# L = cholesky_crout(A)



# print(L_r,L_c,L_data,L_nnz)
# print(Ln_r,Ln_c,Ln_data,Ln_nnz)
# print(Ln_c)

# for i in range(L_nnz):
#     if (np.abs(L_data[i]-Ln_data[i])>1e-10):
#         print(i,L_data[i],Ln_data[i])

# for i in range(L_nnz):
#     if (L_r[i]!=Ln_r[i]):
#         print(i,L_r[i],Ln_r[i])

for i in range(n):
    for j in range(n):
        if (np.abs(L_par[i][j]-L_par2[i][j])>1e-10):
            print(i,j,L_par[i][j],L_par2[i][j])
            print("Error")
            break


if (False):
    start = time.time()
    square_process = Process(target=calculate_square)
    square2_process = Process(target=calculate_square)
    cube_process = Process(target=calculate_cube)
    cube2_process = Process(target=calculate_cube)
    square_process.start()
    cube_process.start()
    square2_process.start()
    cube2_process.start()
    square_process.join()
    cube_process.join()
    square2_process.join()
    cube2_process.join()
    end = time.time()
    print("Time multiprocessing", end - start)


    start=time.time()
    calculate_square()
    calculate_square()
    calculate_cube()
    calculate_cube()
    end = time.time()
    print("Time normal", end-start)


