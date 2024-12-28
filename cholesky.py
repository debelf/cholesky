import numpy as np
import time
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.sparse as sp

def differences(nx,nz):
    L=1
    H=1
    dx=1/nx
    dz=1/nz
    nx+=1
    nz+=1
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

    return A.toarray()

def forward_substitution_coo(L_data,L_r,L_c,L_nnz,b):
    #&&&&&&&&&&&&&&&&&&&&&&&&&
    # REANALYSER
    #&&&&&&&&&&&&&&&&&&&&&&&&&
    x=np.zeros(len(b))
    current_r=0
    for i in range(len(b)):
        sum=0
        val=0
        for index in range(current_r,L_nnz):
            if (L_c[index]==i):
                val=L_data[index]
                current_r=index+1
                break
            sum+=L_data[index]*x[int(L_c[index])]
        x[i]=(b[i]-sum)/val
    return x

def backward_substitution_coo(L_data,L_r,L_c,L_nnz,b):
    #&&&&&&&&&&&&&&&&&&&&&&&&&
    # REANALYSER
    #&&&&&&&&&&&&&&&&&&&&&&&&&
    x=np.zeros(len(b))
    current_r=L_nnz-1
    for i in range(len(b)-1,-1,-1):
        sum=0
        val=0
        for index in range(current_r,-1,-1):
            if (L_r[index]==i):
                if (L_c[index]==i):
                    val=L_data[index]
                    current_r=index-1
                    break
                sum+=L_data[index]*x[int(L_c[index])]
            if (L_r[index]<i):
                break
        x[i]=(b[i]-sum)/val
    return x

def find_band(A_r,A_c):
    n=len(A_r)
    band=0
    for i in range(n):
        if (A_r[i]-A_c[i]>band):
            band=A_r[i]-A_c[i]
    return band

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
    L_r=np.zeros(n*n)
    L_c=np.zeros(n*n)
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

def cholesky_crout2_coo(n, A):
    "ajouter des breaks"
    L_nnz=0
    L_data=np.zeros(n*n)
    L_r=np.zeros(n*n)
    L_c=np.zeros(n*n)

    for j in range(n):
        sum=0
        current_j=0
        for index in range(L_nnz):
            if (L_r[index]==j and L_c[index]<j):
                if (current_j==0):
                    current_j=index 
                sum+=L_data[index]*L_data[index]
        calc=np.sqrt(A[j][j]-sum)
        if (calc>0):
            L_data[L_nnz]=calc
            L_r[L_nnz]=j
            L_c[L_nnz]=j
            L_nnz+=1
        for i in range(j+1,n):
            sum=0
            for index2 in range(current_j,L_nnz):
                if (L_r[index2]==j):
                    for index in range(index2,L_nnz):
                        if (L_r[index]>=i):
                            if (L_r[index]==i):
                                if (L_c[index2]==L_c[index]):
                                    sum+=L_data[index]*L_data[index2]
                                    # print(colored("in","red"))
                            elif (L_r[index]>i):
                                break
                # print(i, "j ",int(L_r[index2]), j)
                # if (L_r[index2]>j):
                #     print(colored("break","magenta"))
            L_den=0
            for index2 in range(current_j,L_nnz):
                if (L_r[index2]==j and L_c[index2]==j):
                    L_den=L_data[index2]
                    break
            calc=(A[i][j]-sum)/L_den
            if (np.abs(calc)>0):
                L_data[L_nnz]=calc
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_crout3_coo(n, A):
    "mettre L_den hors de la boucle+1 break" 
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

def cholesky_banachiewicz(A):
    # row by row
    n=A.shape[0]
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            sum=0
            for k in range(j):
                sum+= L[i][k]*L[j][k]
            if (i==j):
                L[i][j]=np.sqrt(A[i][i]-sum)
            else:
                L[i][j]=(A[i][j]-sum)/L[j][j]
    return L

def cholesky_banachiewicz_coo(n, A):
    L_nnz=0
    L_data=np.zeros(n*n)
    L_r=np.zeros(n*n)
    L_c=np.zeros(n*n)
    for i in range(n):
        for j in range(i+1):
            sum=0
            for index in range(L_nnz):
                if (L_r[index]==i):
                    for index2 in range(L_nnz):
                        if (L_r[index2]==j):
                            if (L_c[index]==L_c[index2]):
                                sum+=L_data[index]*L_data[index2]
            if (i==j):
                calc=np.sqrt(A[i][i]-sum)
                if (calc>0):
                    L_data[L_nnz]=calc
                    L_r[L_nnz]=i
                    L_c[L_nnz]=j
                    L_nnz+=1
            else:
                for index in range(L_nnz):
                    if (L_r[index]==j and L_c[index]==j):
                        calc=(A[i][j]-sum)/L_data[index]
                        if (calc!=0):
                            L_data[L_nnz]=calc
                            L_r[L_nnz]=i
                            L_c[L_nnz]=j
                            L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz2_coo(n, A):
    "break"
    L_nnz=0
    L_data=np.zeros(n*n)
    L_r=np.zeros(n*n)
    L_c=np.zeros(n*n)
    for i in range(n):
        current_j=0
        for j in range(i+1):
            sum=0
            for index in range(L_nnz):
                if (L_r[index]==i):
                    for index2 in range(current_j,L_nnz):
                        if (L_r[index2]>=j):
                            if (L_r[index2]==j):
                                if (L_c[index]==L_c[index2]):
                                    sum+=L_data[index]*L_data[index2]
                            else:
                                break   
            if (i==j):
                calc=np.sqrt(A[i][i]-sum)
                if (calc>0):
                    L_data[L_nnz]=calc
                    L_r[L_nnz]=i
                    L_c[L_nnz]=j
                    L_nnz+=1
            else:
                for index in range(current_j,L_nnz):
                    if (L_r[index]==j and L_c[index]==j):
                        calc=(A[i][j]-sum)/L_data[index]
                        if (calc!=0):
                            L_data[L_nnz]=calc
                            L_r[L_nnz]=i
                            L_c[L_nnz]=j
                            L_nnz+=1
                        current_j=index
                        break
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz3_coo(n, A):
    "cut boucle index i"
    L_nnz=0
    L_data=np.zeros(n*n)
    L_r=np.zeros(n*n)
    L_c=np.zeros(n*n)
    for i in range(n):
        current_j=0
        current_i=L_nnz
        for j in range(i+1):
            sum=0
            # print(colored("Start","red"))
            for index in range(current_i,L_nnz):
                if (L_r[index]==i):
                    # print("ici")
                    for index2 in range(current_j,L_nnz):
                        if (L_r[index2]>=j):
                            if (L_r[index2]==j):
                                if (L_c[index]==L_c[index2]):
                                    sum+=L_data[index]*L_data[index2]
                            else:
                                break
                    current_j=min(current_j,index)
            # print(i,j,sum)
            if (i==j):
                calc=np.sqrt(A[i][i]-sum)
                if (calc>0):
                    L_data[L_nnz]=calc
                    L_r[L_nnz]=i
                    L_c[L_nnz]=j
                    L_nnz+=1
            else:
                for index in range(current_j,L_nnz):
                    if (L_r[index]==j and L_c[index]==j):
                        calc=(A[i][j]-sum)/L_data[index]
                        if (calc!=0):
                            L_data[L_nnz]=calc
                            L_r[L_nnz]=i
                            L_c[L_nnz]=j
                            L_nnz+=1
                        current_j=index
                        break
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz4_coo(n, A):
    "understand index and rewrite correctly"
    L_nnz=0
    L_data=np.zeros(50*n)
    L_r=np.zeros(50*n)
    L_c=np.zeros(50*n)
    begin_r=np.zeros(n,dtype=int)+-1
    begin_r[0]=0
    for i in range(n):
        begin_r[i]=L_nnz
        for j in range(i):
            sum=0
            for index in range(begin_r[i],L_nnz):
                for index2 in range(begin_r[j],begin_r[j+1]):
                    if (L_c[index]==L_c[index2]):
                        sum+=L_data[index]*L_data[index2]
            index=begin_r[j+1]-1
            calc=(A[i][j]-sum)/L_data[index]
            if (calc!=0):
                if (begin_r[i]==-1):
                    begin_r[i]=L_nnz
                L_data[L_nnz]=calc
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
        sum=0
        for index in range(begin_r[i],L_nnz):
            sum+=L_data[index]*L_data[index]
        calc=np.sqrt(A[i][i]-sum)
        if (calc>0):
            L_data[L_nnz]=calc
            L_r[L_nnz]=i
            L_c[L_nnz]=i
            L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz5_coo(n, A):
    "small change"
    L_nnz=0
    L_data=np.zeros(50*n)
    L_r=np.zeros(50*n,dtype=int)
    L_c=np.zeros(50*n,dtype=int)
    begin_r=np.zeros(n,dtype=int)
    for i in range(n):
        begin_r[i]=L_nnz
        for j in range(i):
            sum=0
            for index in range(begin_r[i],L_nnz):
                k=L_c[index]
                for index2 in range(begin_r[j],begin_r[j+1]):
                    if (k==L_c[index2]):
                        sum+=L_data[index]*L_data[index2]
            calc=(A[i][j]-sum)/L_data[begin_r[j+1]-1]
            if (calc!=0):
                L_data[L_nnz]=calc
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
        sum=0
        for index in range(begin_r[i],L_nnz):
            sum+=L_data[index]*L_data[index]
        calc=np.sqrt(A[i][i]-sum)
        if (calc>0):
            L_data[L_nnz]=calc
            L_r[L_nnz]=i
            L_c[L_nnz]=i
            L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz6_coo(n, A_data, A_r, A_c, A_nnz):
    "A is also coo + small changes"
    L_nnz=0
    L_data=np.zeros(100*n)
    L_r=np.zeros(100*n,dtype=int)
    L_c=np.zeros(100*n,dtype=int)
    begin_r=np.zeros(n,dtype=int)
    # print(A_r,A_c)
    current_r=0
    for i in range(n):
        begin_r[i]=L_nnz
        current_i=L_nnz
        # current_j_next=0
        for j in range(i):
            sum=0
            # current_j=current_j_next
            # current_j_next=begin_r[j+1]
            for index in range(current_i,L_nnz):
                k=L_c[index]
                for index2 in range(begin_r[j],begin_r[j+1]):
                    if (k==L_c[index2]):
                        sum+=L_data[index]*L_data[index2]
            
            val=0
            for current in range(current_r,A_nnz):
                # print(current_r, current, A_r[current], A_c[current], i ,j )
                # print(A_r[current], A_c[current], i ,j )
                # if (A_r[current]<=i):
                    if (A_r[current]==i):
                        if (A_c[current]>j):
                            break
                        if (A_c[current]==j):
                            val=A_data[current]
                            current_r=current+1
                            break
                # else:
                #     print(colored("ici","red"))
                #     break
            if (sum!=0 or val!=0):
                L_data[L_nnz]=(val-sum)/L_data[begin_r[j+1]-1]
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
        sum=0
        for index in range(current_i,L_nnz):
            sum+=L_data[index]*L_data[index]
        val=0
        for current in range(current_r,A_nnz):
            # if (A_r[current]>=i):
                if (A_r[current]==i):
                    if (A_c[current]==i):
                        val=A_data[current]
                        current_r=current+1
                        break
                # else:
                #     print(colored("ici","magenta"))
                #     break
        L_data[L_nnz]=np.sqrt(val-sum)
        L_r[L_nnz]=i
        L_c[L_nnz]=i
        L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz7_coo(n, A_data, A_r, A_c, A_nnz):
    "band computation + one break + modif double for index index2"
    # band = find_band(A_r,A_c)
    # print(band)
    L_nnz=0
    L_data=np.zeros(100*n)
    L_r=np.zeros(100*n,dtype=int)
    L_c=np.zeros(100*n,dtype=int)
    begin_r=np.zeros(n,dtype=int)
    # print(A_r,A_c)
    current_r=0
    for i in range(n):
        begin_r[i]=L_nnz
        current_i=L_nnz
        # current_j_next=0
        # print("i,curent_i",i-band)
        for current in range(current_r,A_nnz):
            if (A_r[current]==i):
                # print(current_r,current)
                current_r=current
                band=i-A_c[current]
                break

        for j in range(max(0,i-band),i):
            sum=0

            current_j=begin_r[j]
            current_j_next=begin_r[j+1]
            for index in range(current_i,L_nnz):
                k=L_c[index]
                for index2 in range(current_j,current_j_next):
                    if (k==L_c[index2]):
                        sum+=L_data[index]*L_data[index2]
                        current_j=index2+1
                        break
            
            # next_index=current_i
            # for index2 in range(begin_r[j],begin_r[j+1]):
            #     k=L_c[index2]
            #     for index in range(next_index,L_nnz):
            #         if (L_c[index]==k):
            #             next_index=index+1
            #             sum+=L_data[index]*L_data[index2]
            #             break
            #         if (L_c[index]>k):
            #             next_index=index
            #             break

            val=0
            for current in range(current_r,A_nnz):
                # print(current_r, current, A_r[current], A_c[current], i ,j )
                # print(A_r[current], A_c[current], i ,j )
                # if (A_r[current]<=i):
                    if (A_r[current]==i):
                        if (A_c[current]>j):
                            break
                        if (A_c[current]==j):
                            val=A_data[current]
                            current_r=current+1
                            break
                # else:
                #     print(colored("ici","red"))
                #     break
            if (sum!=0 or val!=0):
                L_data[L_nnz]=(val-sum)/L_data[current_j_next-1]
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
        sum=0
        for index in range(current_i,L_nnz):
            sum+=L_data[index]*L_data[index]
        val=0
        for current in range(current_r,A_nnz):
            # if (A_r[current]>=i):
                if (A_r[current]==i):
                    if (A_c[current]==i):
                        val=A_data[current]
                        current_r=current+1
                        break
                # else:
                #     print(colored("ici","magenta"))
                #     break
        L_data[L_nnz]=np.sqrt(val-sum)
        L_r[L_nnz]=i
        L_c[L_nnz]=i
        L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def cholesky_banachiewicz8_coo(n, A_data, A_r, A_c, A_nnz):
    "clean"
    L_nnz=0
    L_data=np.zeros(100*n)
    L_r=np.zeros(100*n,dtype=int)
    L_c=np.zeros(100*n,dtype=int)
    begin_r=np.zeros(n,dtype=int)
    current_r=0
    for i in range(n):
        begin_r[i]=L_nnz
        current_i=L_nnz
        for current in range(current_r,A_nnz):
            if (A_r[current]==i):
                current_r=current
                band=i-A_c[current]
                break

        for j in range(max(0,i-band),i):
            sum=0
            current_j=begin_r[j]
            current_j_next=begin_r[j+1]
            for index in range(current_i,L_nnz):
                k=L_c[index]
                for index2 in range(current_j,current_j_next):
                    if (k==L_c[index2]):
                        sum+=L_data[index]*L_data[index2]
                        current_j=index2+1
                        break
            val=0
            for current in range(current_r,A_nnz):
                if (A_r[current]==i):
                    if (A_c[current]>j):
                        break
                    if (A_c[current]==j):
                        val=A_data[current]
                        current_r=current+1
                        break
            if (sum!=0 or val!=0):
                L_data[L_nnz]=(val-sum)/L_data[current_j_next-1]
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
        sum=0
        for index in range(current_i,L_nnz):
            sum+=L_data[index]*L_data[index]
        val=0
        for current in range(current_r,A_nnz):
            if (A_r[current]==i):
                if (A_c[current]==i):
                    val=A_data[current]
                    current_r=current+1
                    break
        L_data[L_nnz]=np.sqrt(val-sum)
        L_r[L_nnz]=i
        L_c[L_nnz]=i
        L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def coo_to_dense(L_data,L_r,L_c,n):
    L=np.zeros((n,n))
    for i in range(len(L_data)):
        L[int(L_r[i])][int(L_c[i])]=L_data[i]
    return L

def dense_to_coo(L):
    n=L.shape[0]
    L_nnz=0
    L_data=np.zeros(20*n)
    L_r=np.zeros(20*n, dtype=int)
    L_c=np.zeros(20*n, dtype=int)
    for i in range(n):
        for j in range(n):
            if (L[i][j]!=0):
                L_data[L_nnz]=L[i][j]
                L_r[L_nnz]=i
                L_c[L_nnz]=j
                L_nnz+=1
    return L_data[:L_nnz],L_r[:L_nnz],L_c[:L_nnz],L_nnz

def transpose_coo2(L_data,L_r,L_c,L_nnz,n):
    Lt_data = L_data
    Lt_r = L_c
    Lt_c = L_r
    Lt_r,Lt_c,Lt_data = counting_sort(Lt_r,Lt_c,Lt_data, 0, n)
    return Lt_data, Lt_r, Lt_c, L_nnz

def counting_sort(L_r,L_c,L_data, index, max_value):
    # index = 0 pour x, index = 1 pour y
    count = np.zeros(max_value + 1,dtype=int)
    L_r_new = np.zeros(len(L_r), dtype=int)
    L_c_new = np.zeros(len(L_c), dtype=int)
    L_data_new = np.zeros(len(L_c))

    points = np.array([L_r, L_c, L_data]).T

    # Compter les occurrences des valeurs
    for point in points:
        count[int(point[index])] += 1

    # Cumuler les occurrences
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Placer les éléments dans leur position correcte (en ordre inverse pour stabilité)
    for point in reversed(points):
        current = int(point[index])
        count[current] -= 1
        L_r_new[count[current]] = point[0]
        L_c_new[count[current]] = point[1]
        L_data_new[count[current]] = point[2]

    return L_r_new, L_c_new, L_data_new

def approximate_minimum_degree(matrix):
    n = matrix.shape[0]
    degrees = np.array(matrix.sum(axis=1)).flatten()
    visited = np.zeros(n, dtype=bool)
    order = []

    for _ in range(n):
        # Find the unvisited node with the smallest degree
        min_degree_node = np.argmin(np.where(visited, np.inf, degrees))
        visited[min_degree_node] = True
        order.append(min_degree_node)

        # Update degrees of remaining nodes
        for neighbor in np.where(matrix[min_degree_node].toarray().flatten() > 0)[0]:
            if not visited[neighbor]:
                degrees[neighbor] -= 1

    return order

# A=np.array([[10,2,3,-5],
#             [2,60,1,1],
#             [3,1,13,2],
#             [-5,1,2,9]])

# n=5
# A=np.random.random((n,n))
# At=A.T
# A=A+At
# A=np.dot(A,At)
# A=A+(n*3)*np.eye(n)

# A=np.array([[11.75113208, 1.7032331, 1.3917452, 0.95079053, 2.83870064],
# [ 2.32904274, 13.81313495,  3.07198358,  1.45988658,  4.65307944],
# [ 2.10075773,  2.91064754, 12.48252549,  1.20796037,  4.38643729],
# [ 1.29993784,  1.64249814,  1.31401536, 10.75826002,  2.50977833],
# [ 2.68688934,  3.45990689,  2.94683839, 1.53770277, 15.24577252]])

# seed = 42
# rng = np.random.default_rng(seed)
# np.random.seed(seed)
# n=10000
# random_matrix = sp.random(n, n, density=1.05/n, format='csr', data_rvs=lambda n: np.random.choice([0, 1], size=n), random_state=rng)
# random_matrix.setdiag(7)
# A=random_matrix.toarray()
# A+=A.T

n=50
A=differences(n,n)
n=A.shape[0]

from scipy.sparse.csgraph import reverse_cuthill_mckee
start=time.time()
order=reverse_cuthill_mckee(sp.csr_matrix(A), symmetric_mode=True)
end=time.time()
print(colored("Time rcmk : {:f}".format(end-start), "red"))

# start=time.time()
# order=approximate_minimum_degree(sp.csr_matrix(A))
# end=time.time()
# print(colored("Time amd : {:f}".format(end-start), "red"))

A_reordered = A[order, :][:, order]
A_data,A_r,A_c,A_nnz=dense_to_coo(A_reordered)
print("A_nnz : ",A_nnz)

b=np.ones(n)
plt.spy(A_reordered, markersize=1)
plt.show()

# Compare solver to numpy
if (False):
    # TEST AVEC SOLVER
    start_all=time.time()   
    start=time.time()
    L_data,L_r,L_c,L_nnz=cholesky_banachiewicz8_coo(n,A_data,A_r,A_c,A_nnz)
    print(time.time()-start)
    start=time.time()
    y=forward_substitution_coo(L_data,L_r,L_c,L_nnz,b)
    print(time.time()-start)
    start=time.time()
    Lt_data,Lt_r,Lt_c,Lt_nnz=transpose_coo2(L_data,L_r,L_c,L_nnz,n)
    print(time.time()-start)
    start=time.time()
    x=backward_substitution_coo(Lt_data,Lt_r,Lt_c,Lt_nnz,y)
    print(time.time()-start)
    end=time.time()
    print(colored("Time bana_coo : {:f}".format(end-start_all), "magenta"))

    start=time.time()
    # x_real=np.linalg.solve(coo_to_dense(L_data,L_r,L_c,len(A)),b)
    x_real=np.linalg.solve(A_reordered,b) 
    end=time.time()
    print(colored("Time numpy : {:f}".format(end-start), "magenta"))

    for i in range(n):
        if (np.abs(x[i]-x_real[i])>1e-10):
            print(colored("Error","red"))
            break

    # print("x_comp: ")
    # print(x)
    # print("x_real: ")
    # print(x_real)
    exit()

# Complexity test
if (False):
    n_list=np.array([10,20,30,40,50,60,70,80,90,100])
    nnz_list=np.zeros(len(n_list))
    time_list=np.zeros(len(n_list))
    current=0
    for n in n_list:
        A=differences(n,n)
        b=np.ones(n)
        nnz_list[current]=A.shape[0]
        start=time.time()
        A_data,A_r,A_c,A_nnz=dense_to_coo(A)
        L_data,L_r,L_c,L_nnz=cholesky_banachiewicz8_coo(n,A_data,A_r,A_c,A_nnz)
        y=forward_substitution_coo(L_data,L_r,L_c,L_nnz,b)
        Lt_data,Lt_r,Lt_c,Lt_nnz=transpose_coo2(L_data,L_r,L_c,L_nnz,n)
        x=backward_substitution_coo(Lt_data,Lt_r,Lt_c,Lt_nnz,y)
        end=time.time()
        time_list[current]=end-start
        print(colored("Time bana_coo : {:f}".format(end-start), "magenta"))
        current+=1
        
    plt.plot(nnz_list,time_list,"-o", color="darkred", label="cholesky")
    plt.plot(nnz_list,[i**(3/2) for i in nnz_list],"-o", color="navy", label=r"$\mathcal{O}(n_z^{3/2})$")
    plt.yscale("log")
    plt.xscale("log")   
    plt.legend()
    plt.grid(alpha=0.7)
    plt.savefig("complexity.pdf")
    plt.show()
    exit()

# Profiling factorization
if (True):
    start=time.time()
    import line_profiler
    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(cholesky_banachiewicz8_coo)
    L_data,L_r,L_c,L_nnz=lp_wrapper(n,A_data,A_r,A_c,A_nnz)
    lp.print_stats()
    end=time.time()
    print(colored("Time : {:f}".format(end-start), "red"))
    exit()

# Profiling forward and backward substitution
if (False):
    start_all=time.time()   
    start=time.time()
    L_data,L_r,L_c,L_nnz=cholesky_banachiewicz6_coo(n,A_data,A_r,A_c,A_nnz)
    print(time.time()-start)
    start=time.time()

    import line_profiler
    start=time.time()
    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(forward_substitution_coo)
    y=lp_wrapper(L_data,L_r,L_c,L_nnz,b)
    lp.print_stats()
    print(colored(time.time()-start,"red"))

    start=time.time()
    Lt_data,Lt_r,Lt_c,Lt_nnz=transpose_coo(L_data,L_r,L_c,L_nnz,n)
    print(time.time()-start)

    start=time.time()
    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(backward_substitution_coo)
    x=lp_wrapper(Lt_data,Lt_r,Lt_c,Lt_nnz,y)
    lp.print_stats()
    print(colored(time.time()-start,"red"))
    end=time.time()
    print(colored("Time bana_coo : {:f}".format(end-start_all), "magenta"))

    start=time.time()
    x_real=np.linalg.solve(A_reordered,b) 
    end=time.time()
    print(colored("Time numpy : {:f}".format(end-start), "magenta"))

    print("x_comp: ")
    print(x)
    print("x_real: ")
    print(x_real)

    for i in range(n):
        if (np.abs(x[i]-x_real[i])>1e-10):
            print(colored("Error","red"))
            break

    exit()
 
# TEST cholesky_banachiewicz(A) et cholesky_crout
if (False):
    Lb=cholesky_banachiewicz(A)
    Lc=cholesky_crout(A)
    Ln=np.linalg.cholesky(A)

    print(colored(colored("Check start","green")))
    for i in range(n):
        for j in range(n):
            if (np.abs(Lb[i][j]-Ln[i][j])>1e-10):
                print(colored("Matrix Lb not equal to Ln","red"))
                break
    print(colored(colored("Check over","green")))
    
    print(colored(colored("Check start","green")))
    for i in range(n):
        for j in range(n):
            if (np.abs(Lc[i][j]-Ln[i][j])>1e-10):
                print(colored("Matrix Lc not equal to Ln","red"))
                break
    print(colored(colored("Check over","green")))

def csr(A):
    m=A.shape[0]
    n=A.shape[1]
    nnz=0
    for i in range(m):
        for j in range(n):
            if(A[i][j]!=0):
                nnz+=1
    data=np.zeros(nnz)
    column_index=np.zeros(nnz)
    row_index=np.zeros(m+1)
    current=0
    for i in range(m):
        rows=0
        for j in range(n):
            if (A[i][j]!=0):
                data[current]=A[i][j]
                column_index[current]=j
                rows+=1
                current+=1
        row_index[i+1]=row_index[i]+rows 
    return nnz,data,column_index,row_index

# TEST CSR
if (True):
    A=np.array([[5,0,0,0],
                [0,8,0,0],
                [0,0,3,0],
                [0,6,0,0]])
    print(csr(A))
    print(colored("(4, array([5., 8., 3., 6.]), array([0., 1., 2., 1.]), array([0., 1., 2., 3., 4.]))", "green"))

    A=np.array([[10,20,0,0,0,0],
                [0,30,0,40,0,0],
                [0,0,50,60,70,0],
                [0,0,0,0,0,80]])
    print(csr(A))
    print(colored("(8, array([10., 20., 30., 40., 50., 60., 70., 80.]), array([0., 1., 1., 3., 2., 3., 4., 5.]), array([0., 2., 4., 7., 8.]))", "green"))