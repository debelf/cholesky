# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <math.h>
# include "matrix.h"
# pragma GCC optimize("O3")
# define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
# define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

void print_test(int i){
    printf("STARTING PROCESS %d\n", i);
}

void gauss_seidel_coo(matrix_coo *A, vector *b, vector *x, int step_max) {
    int n = b->size;
    vector *x_new = vector_create(n);

    double epsilon = 1;
    int step = 0;

    // Pointeur sur le début des éléments pour chaque ligne
    int start = 0;

    while (epsilon > 1e-10 && step < step_max) {
        for (int i = 0; i < n; i++) {
            x_new->data[i] = x->data[i];  // Initialize x_new
        }

        for (int i = 0; i < n; i++) {
            double sum_ = 0.0;
            double diag = 0.0;

            // Parcourir uniquement les non-zéros de la ligne actuelle (grâce au tri)
            while (start < A->nnz && A->ir[start] == i) {
                int col = A->ic[start];
                if (col == i) {
                    diag = A->data[start];  // Élément diagonal
                } else {
                    sum_ += A->data[start] * x_new->data[col];  // Contribution hors-diagonale
                }
                start++;
            }

            // Mettre à jour x_new[i] en utilisant la formule
            if (diag != 0) {
                x_new->data[i] = (b->data[i] - sum_) / diag;
            }
        }

        // Réinitialiser start pour parcourir les éléments à nouveau
        start = 0;

        // Calculer la norme et mettre à jour x
        epsilon = 0;
        for (int i = 0; i < n; i++) {
            double diff = x_new->data[i] - x->data[i];
            epsilon += diff * diff;
            x->data[i] = x_new->data[i];
        }
        epsilon = sqrt(epsilon);
        if (epsilon < 1e-10) {
            printf("Step %d\n", step);
        }
        step++;
        // printf("Step: %d\n", step);
    }
    printf("Step %d\n", step);
    vector_destroy(x_new);
}

void conjugate_gradient_coo(matrix_coo *A, vector *b, vector *x, int step_max){
    int n=b->size;
    vector *r=vector_create(n);
    vector *p=vector_create(n);
    vector *temp=matrix_vector_multiplication_coo(A,x);
    double res_old=0;
    for (int i = 0; i < n; i++) {
        r->data[i]=b->data[i]-temp->data[i];
        p->data[i]=r->data[i];
        res_old+=r->data[i]*r->data[i];
    }

    int k=0;
    while (k<step_max){
        vector *Ap=matrix_vector_multiplication_coo(A,p);
        double p_dot_Ap=0;
        for (int i = 0; i < n; i++) {
            p_dot_Ap+=p->data[i]*Ap->data[i];
        }
        double alpha=res_old/p_dot_Ap;
        for (int i = 0; i < n; i++) {
            x->data[i]+=alpha*p->data[i];
            r->data[i]-=alpha*Ap->data[i];
        }
        double beta;
        double res_new=0;
        for (int i = 0; i < n; i++) {
            res_new+=r->data[i]*r->data[i];
        }
        beta=res_new/res_old;
        for (int i = 0; i < n; i++) {
            p->data[i]=r->data[i]+beta*p->data[i];
        }
        res_old=res_new;
        if (res_new<1e-10){
            break;
        }
        k++;
        vector_destroy(Ap);
    }
    printf("Step %d\n", k);
    vector_destroy(r);
    vector_destroy(p);
    vector_destroy(temp);
}

int band_computation(int nnz, int *A_r, int *A_c){
    int max=0;
    for (int i = 0; i < nnz; i++)
        max=MAX(max,abs(A_r[i]-A_c[i]));
    return max;
}

matrix_coo* cholesky_banachiewicz8_coo(int n, matrix_coo *A){
    int A_nnz=A->nnz;
    int *A_r=A->ir;
    int *A_c=A->ic;
    double *A_data=A->data;
    int band=band_computation(A_nnz,A_r,A_c);
    matrix_coo *L=matrix_coo_create((band+1)*n,n,n);
    int L_nnz=0;
    double *L_data=L->data;
    int *L_r=L->ir;
    int *L_c=L->ic;
    int *begin_r=(int*) malloc(n*sizeof(int));
    int current_r=0;
    int current_i,current_j,current_j_next;
    double sum,val;
    for (int i=0; i<n;i++){
        begin_r[i]=L_nnz;
        current_i=L_nnz;
        for (int current=0;current<A_nnz;current++){
            if (A_r[current]==i){
                current_r=current;
                band=i-A_c[current];
                break;
            }
        }
        for (int j=MAX(0,i-band);j<i;j++){
            sum=0;
            current_j=begin_r[j];
            current_j_next=begin_r[j+1];
            for (int index=current_i;index<L_nnz;index++){
                int k=L_c[index];
                for (int index2=current_j;index2<current_j_next;index2++){
                    if (k==L_c[index2]){
                        sum+=L_data[index]*L_data[index2];
                        current_j=index2+1;
                        break;
                    }
                }
            }
            val=0;
            for (int current=current_r; current<A_nnz;current++){
                if (A_r[current]==i){
                    if (A_c[current]>j)
                        break;
                    if (A_c[current]==j){
                        val=A_data[current];
                        current_r=current+1;
                        break;
                    }
                }
            }
            if (sum!=0 || val!=0){
                L_data[L_nnz]=(val-sum)/L_data[current_j_next-1];
                L_r[L_nnz]=i;
                L_c[L_nnz]=j;
                L_nnz+=1;
            }
        }
        sum=0;
        for (int index=current_i;index<L_nnz;index++)
            sum+=L_data[index]*L_data[index];
        val=0;
        for (int current=current_r; current<A_nnz;current++){
            if (A_r[current]==i){
                if (A_c[current]==i){
                    val=A_data[current];
                    current_r=current+1;
                    break;
                }
            }
        }
        L_data[L_nnz]=sqrt(val-sum);
        L_r[L_nnz]=i;
        L_c[L_nnz]=i;
        L_nnz+=1;
    }
    free(begin_r);
    L->nnz=L_nnz;
    return L;
}

vector* forward_substitution_coo(matrix_coo *L, vector *b){
    int L_nnz=L->nnz;
    double *L_data=L->data;
    int *L_c=L->ic;
    int n=L->n;
    vector *x=vector_create(n);
    int current_r=0;

    for (int i=0; i<n;i++){
        double sum=0;
        double val=0;
        for (int index=current_r;index<L_nnz;index++){
            if (L_c[index]==i){
                val=L_data[index];
                current_r=index+1;
                break;
            }
            sum+=L_data[index]*x->data[L_c[index]];
        }
        x->data[i]=(b->data[i]-sum)/val;
    }
    return x;
}

vector* backward_substitution_coo(matrix_coo *L, vector *b){
    int L_nnz=L->nnz;
    double *L_data=L->data;
    int *L_r=L->ir;
    int *L_c=L->ic;
    int n=L->n;
    vector *x=vector_create(n);
    int current_r=L_nnz-1;

    for (int i=n-1; i>=0;i--){
        double sum=0;
        double val=0;
        for (int index=current_r;index>=0;index--){
            if (L_r[index]==i){
                if (L_c[index]==i){
                    val=L_data[index];
                    current_r=index-1;
                    break;
                }
                sum+=L_data[index]*x->data[L_c[index]];
            }
            if (L_r[index]<i){
                break;
            }
        }
        x->data[i]=(b->data[i]-sum)/val;
    }
    return x;
}

matrix_coo* counting_sort(int L_nnz, int *L_r, int *L_c, double *L_data, int n){
    // trie sur les rangées
    matrix_coo *Lt = matrix_coo_create(L_nnz,n,n);
    int *count=(int*) calloc(n+1, sizeof(int));
    for (int i = 0; i < L_nnz; i++)
        count[L_r[i]] += 1;
    for (int i = 1; i < n+1; i++)
        count[i] += count[i - 1];
    // en ordre inverse pour la stabilité, pour pas modifier le tri sur les colonnes
    for (int i = L_nnz-1; i >= 0; i--){
        int current = L_r[i];
        count[current] -= 1;
        Lt->ir[count[current]] = L_r[i];
        Lt->ic[count[current]] = L_c[i];
        Lt->data[count[current]] = L_data[i];
    }
    free(count);
    return Lt;
}

matrix_coo* transpose_coo2(matrix_coo *L){
    int L_nnz=L->nnz;
    double *L_data=L->data;
    int *L_r=L->ic; // L_r -> L_c
    int *L_c=L->ir; // L_c -> L_r
    int n=L->n;
    return counting_sort(L_nnz, L_r, L_c, L_data, n);
}

void run(int nnz, int m, int n, int *rows, int *cols, double *data, double *b, double *x) {
    matrix_coo *A_coo=matrix_coo_create(nnz,m,n);
    for (int i = 0; i < nnz; i++) {
        A_coo->ir[i]=rows[i];
        A_coo->ic[i]=cols[i];
        A_coo->data[i]=data[i];
    }
    vector *b_vec = vector_create(n);
    for (int i = 0; i < n; i++)
        b_vec->data[i]=b[i];
    
    int flag_direct=0;
    // solver
    if (flag_direct==0){
        matrix_coo *L=cholesky_banachiewicz8_coo(n,A_coo);
        matrix_coo *Lt=transpose_coo2(L);
        vector *y=forward_substitution_coo(L,b_vec);
        vector *x_vec=backward_substitution_coo(Lt,y);
        for (int i = 0; i < n; i++)
            x[i]=x_vec->data[i];
        matrix_coo_destroy(A_coo);
        matrix_coo_destroy(L);
        matrix_coo_destroy(Lt);
        vector_destroy(b_vec);
        vector_destroy(y);
        vector_destroy(x_vec);
    } else{
        vector *x_vec = vector_create(n);
        conjugate_gradient_coo(A_coo,b_vec,x_vec,10000); 
        for (int i = 0; i < n; i++)
            x[i]=x_vec->data[i];
    }
}


