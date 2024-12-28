# ifndef MATRIX_H
# define MATRIX_H


typedef struct {
    int rows;
    int cols;
    double **data;
} matrix;

typedef struct {
    int size;
    double *data;
} vector;

typedef struct {
    int nnz;
    int m;
    int n;
    int *ir;
    int *ic;
    double *data;
} matrix_coo;



matrix *matrix_create(int rows, int cols);

void matrix_destroy(matrix *m);

void matrix_print(matrix *m);

matrix *matrix_copy(matrix *m);

matrix *matrix_multiplication(matrix *A, matrix *B);

matrix *matrix_from_txt(int n, char *filename);

vector *vector_create(int size);

void vector_destroy(vector *v);

void vector_print(vector *v);

vector *vector_copy(vector *v);

vector *matrix_vector_multiplication(matrix *A, vector *v);

matrix *matrix_transpose(matrix *A);

vector *vector_from_txt(int n, char *filename);

matrix_coo *matrix_coo_create(int nnz, int m, int n);

matrix_coo *matrix_to_coo(matrix *A);

void matrix_coo_destroy(matrix_coo *m);

vector *matrix_vector_multiplication_coo(matrix_coo *A, vector *v);

matrix_coo *matrix_multiplication_coo(matrix_coo *A, matrix_coo *B);

matrix *matrix_from_coo(matrix_coo *A, int m, int n);

matrix_coo *matrix_transpose_coo(matrix_coo *A);

#endif