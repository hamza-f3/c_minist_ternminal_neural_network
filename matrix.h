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
Matrix	*matrix_sub(Matrix *m1, Matrix *m2);
Matrix	*matrix_dot(Matrix *m1,Matrix *m2);
Matrix	*matrix_transpose(Matrix *m);
#endif 
