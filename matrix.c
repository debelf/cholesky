# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <math.h>
# include "matrix.h"

// matrix constructor --- ATTENTION REDO IT IN PROPEER WAY ---
matrix *matrix_create(int rows, int cols) {
    matrix *m = malloc(sizeof(matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows, sizeof(double *));
    for (int i = 0; i < rows; i++) {
        m->data[i] = calloc(cols, sizeof(double));
    }
    return m;
}

// matrix destructor
void matrix_destroy(matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// matrix print
void matrix_print(matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}

// matrix copy
matrix *matrix_copy(matrix *m) {
    matrix *copy = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            copy->data[i][j] = m->data[i][j];
        }
    }
    return copy;
}

// matrix multiplication
matrix *matrix_multiplication(matrix *A, matrix *B) {
    if (A->cols != B->rows) {
        //fprintf(stderr, "Erreur: Incompatible dimension matrix\n");
        exit(1);
    }
    matrix *R = matrix_create(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            R->data[i][j] = 0;
            for (int k = 0; k < A->cols; k++) {
                R->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return R;
}

matrix *matrix_from_txt(int n, char *filename){
    FILE *file = fopen(filename, "r");
    if (file == NULL){
        //fprintf(stderr, "Erreur: Impossible d'ouvrir le fichier matrix\n");
        exit(1);
    }
    int rows=n, cols=n;
    // matrix format :
    // 1 2 3
    // 4 5 6
    // 7 8 9
    matrix *m = matrix_create(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            fscanf(file, "%lf", &m->data[i][j]);
        }
    }
    return m;
}

// vector constructor
vector *vector_create(int size) {
    vector *v = malloc(sizeof(vector));
    v->size = size;
    v->data = calloc(size, sizeof(double));
    return v;
}

// vector destructor
void vector_destroy(vector *v) {
    free(v->data);
    free(v);
}

// vector print
void vector_print(vector *v) {
    for (int i = 0; i < v->size; i++) {
        printf("%f ", v->data[i]);
    }
    printf("\n");
}

// vector copy
vector *vector_copy(vector *v) {
    vector *copy = vector_create(v->size);
    for (int i = 0; i < v->size; i++) {
        copy->data[i] = v->data[i];
    }
    return copy;
}

vector *matrix_vector_multiplication(matrix *A, vector *v) {
    if (A->cols != v->size) {
        //fprintf(stderr, "Erreur: Incompatible dimension vector\n");
        exit(1);
    }
    vector *R = vector_create(A->rows);
    for (int i = 0; i < A->rows; i++) {
        R->data[i] = 0;
        for (int j = 0; j < A->cols; j++) {
            R->data[i] += A->data[i][j] * v->data[j];
        }
    }
    return R;
}

matrix *matrix_transpose(matrix *A){
    matrix *R = matrix_create(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            R->data[j][i] = A->data[i][j];
        }
    }
    return R;
}

vector *vector_from_txt(int n, char *filename){
    FILE *file = fopen(filename, "r");
    if (file == NULL){
        //fprintf(stderr, "Erreur: Impossible d'ouvrir le fichier vector\n");
        exit(1);
    }
    int size=n;
    // vector format :
    // 1
    // 2
    // 3
    vector *v = vector_create(size);
    for (int i = 0; i < size; i++) {
        fscanf(file, "%lf", &v->data[i]);
    }
    return v;
}

matrix_coo *matrix_coo_create(int nnz, int m, int n) {
    matrix_coo *A = malloc(sizeof(matrix_coo));
    A->nnz = nnz;
    A->m = m;
    A->n = n;
    A->ir = calloc(nnz, sizeof(int));
    A->ic = calloc(nnz, sizeof(int));
    A->data = calloc(nnz, sizeof(double));
    return A;
}

matrix_coo *matrix_to_coo(matrix *A){
    int nnz=0;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            if (A->data[i][j]!=0){
                nnz++;
            }
        }
    }
    matrix_coo *m = matrix_coo_create(nnz,A->rows,A->cols);
    int k=0;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            if (A->data[i][j]!=0){
                m->ir[k]=i;
                m->ic[k]=j;
                m->data[k]=A->data[i][j];
                k++;
            }
        }
    }
    return m;
}

void matrix_coo_destroy(matrix_coo *m){
    free(m->ir);
    free(m->ic);
    free(m->data);
    free(m);
}

vector *matrix_vector_multiplication_coo(matrix_coo *A, vector *v) {
    vector *r = vector_create(A->m);
    // Pour chaque élément non nul dans la matrice
    for (int i = 0; i < A->nnz; i++) {
        int row = A->ir[i];     // Ligne de l'élément
        int col = A->ic[i];     // Colonne de l'élément
        // Produit et addition dans le vecteur résultat
        r->data[row] += A->data[i] * v->data[col];
    }
    return r;
}

matrix_coo *matrix_multiplication_coo(matrix_coo *A, matrix_coo *B) {
    // Utilisation d'une table de hachage pour stocker les résultats (clé = (row, col), valeur = somme des produits)
    int hash_size = A->m * B->n;
    double *hash_data = calloc(hash_size, sizeof(double));
    int *hash_used = calloc(hash_size, sizeof(int)); // Marqueur des entrées utilisées

    // Accumulation des produits
    for (int i = 0; i < A->nnz; i++) {
        int rowA = A->ir[i];
        int colA = A->ic[i];
        double valA = A->data[i];
        int j=0;
        while (j < B->nnz && B->ir[j] < colA) {
            j++;
        }
        // Traiter tous les éléments de B correspondant à colA
        for (int k = j; k < B->nnz && B->ir[k] == colA; k++) {
            int rowC = rowA;
            int colC = B->ic[k];
            double valC = valA * B->data[k];

            // Calcul de l'index de hachage
            int hash_index = rowC * B->n + colC;

            // Ajout au résultat
            hash_data[hash_index] += valC;
            hash_used[hash_index] = 1;
        }
        // for (int j = 0; j < B->nnz; j++) {
        //     if (B->ir[j] == colA) { // Correspondance colonne de A = ligne de B
        //         int rowC = rowA;
        //         int colC = B->ic[j];
        //         double valC = valA * B->data[j];

        //         // Calcul de l'index de hachage
        //         int hash_index = rowC * B->n + colC;

        //         // Ajout au résultat
        //         hash_data[hash_index] += valC;
        //         hash_used[hash_index] = 1;
        //     }
        // }
    }

    // Comptage des éléments non nuls finaux
    int final_nnz = 0;
    for (int i = 0; i < hash_size; i++) {
        if (hash_used[i]) {
            final_nnz++;
        }
    }

    // Construction du résultat
    matrix_coo *C = malloc(sizeof(matrix_coo));
    C->m = A->m;
    C->n = B->n;
    C->nnz = final_nnz;
    C->ir = malloc(final_nnz * sizeof(int));
    C->ic = malloc(final_nnz * sizeof(int));
    C->data = malloc(final_nnz * sizeof(double));

    // Remplissage de la matrice COO
    int idx = 0;
    for (int i = 0; i < hash_size; i++) {
        if (hash_used[i]) {
            C->ir[idx] = i / B->n;        // Ligne
            C->ic[idx] = i % B->n;        // Colonne
            C->data[idx] = hash_data[i]; // Valeur
            idx++;
        }
    }

    // Libération des ressources temporaires
    free(hash_data);
    free(hash_used);

    return C;
}

matrix_coo *matrix_multiplication_coo_old(matrix_coo *A, matrix_coo *B) {
    // Résultat provisoire (tableau dynamique pour accumulation)
    int *res_ir = calloc(A->nnz * B->nnz, sizeof(int));
    int *res_ic = calloc(A->nnz * B->nnz, sizeof(int));
    double *res_data = calloc(A->nnz * B->nnz, sizeof(double));
    int res_nnz = 0;

    // Pour chaque élément non nul de A
    for (int i = 0; i < A->nnz; i++) {
        int rowA = A->ir[i];
        int colA = A->ic[i];
        double valA = A->data[i];

        // Chercher les correspondances dans B
        for (int j = 0; j < B->nnz; j++) {
            int rowB = B->ir[j];
            int colB = B->ic[j];
            double valB = B->data[j];

            // Produit valide si la colonne de A correspond à la ligne de B
            if (colA == rowB) {
                int rowC = rowA;
                int colC = colB;
                double valC = valA * valB;

                // Ajouter le résultat au tableau provisoire
                res_ir[res_nnz] = rowC;
                res_ic[res_nnz] = colC;
                res_data[res_nnz] = valC;
                res_nnz++;
            }
        }
    }

    // Combiner les éléments ayant les mêmes indices (sommation des doublons)
    for (int i = 0; i < res_nnz; i++) {
        for (int j = i + 1; j < res_nnz; j++) {
            if (res_ir[i] == res_ir[j] && res_ic[i] == res_ic[j]) {
                res_data[i] += res_data[j];
                res_ir[j] = -1; // Marqueur pour suppression
            }
        }
    }

    // Compactage des résultats
    int final_nnz = 0;
    for (int i = 0; i < res_nnz; i++) {
        if (res_ir[i] != -1) {
            res_ir[final_nnz] = res_ir[i];
            res_ic[final_nnz] = res_ic[i];
            res_data[final_nnz] = res_data[i];
            final_nnz++;
        }
    }

    // Stocker le résultat dans une nouvelle structure COO
    matrix_coo *C=malloc(sizeof(matrix_coo));
    C->m = A->m;
    C->n = B->n;
    C->nnz = final_nnz;
    C->ir = calloc(final_nnz, sizeof(int));
    C->ic = calloc(final_nnz, sizeof(int));
    C->data = calloc(final_nnz, sizeof(double));
    for (int i = 0; i < final_nnz; i++) {
        C->ir[i] = res_ir[i];
        C->ic[i] = res_ic[i];
        C->data[i] = res_data[i];
    }

    // Libérer les tableaux temporaires
    free(res_ir);
    free(res_ic);
    free(res_data);

    return C;
}

matrix *matrix_from_coo(matrix_coo *A, int m, int n) {
    matrix *p = matrix_create(m, n);
    for (int i = 0; i < A->nnz; i++) {
        p->data[A->ir[i]][A->ic[i]] = A->data[i];
    }
    return p;
}

matrix_coo *matrix_transpose_coo(matrix_coo *A) {
    matrix_coo *At = matrix_coo_create(A->nnz, A->n, A->m);
    for (int i = 0; i < A->nnz; i++) {
        At->ir[i] = A->ic[i];
        At->ic[i] = A->ir[i];
        At->data[i] = A->data[i];
    }
    return At;
}

