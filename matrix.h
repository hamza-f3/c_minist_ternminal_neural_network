#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix *m);

void	matrix_randomize(Matrix *m);
void	matrix_print(Matrix *m);
void	matrix_add(Matrix  *m1, Matrix *m2);
void	matrix_scale(Matrix *m, double n);
Matrix	*matrix_sub(Matrix *m1, Matrix *m2);
Matrix	*matrix_dot(Matrix *m1,Matrix *m2);
Matrix	*matrix_transpose(Matrix *m);
Matrix	*matrix_copy(Matrix *m);
Matrix	*matrix_multiply_elementwise(Matrix *m1, Matrix *m2);
#endif 
