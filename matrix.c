#include "matrix.h"

Matrix *matrix_create(int rows, int cols)
{
	Matrix *m = (Matrix *)malloc(sizeof(Matrix));
	if (!m)
		return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = (double *)calloc(rows *cols,sizeof(double));
	if (!m->data)
	{
		free(m);
		return NULL;
	}
	return m;
}
void	matrix_free(Matrix *m)
{
	if (m)
	{
		if(m->data)
		{
			free(m->data);
		}
		free(m);
	}
}
void matrix_randomize(Matrix *m)
{
	int	i;
	int	total;
	i = 0;
	total = m->rows * m->cols;
	while (i < total)
	{
		m->data[i] = ((double)rand() / RAND_MAX) - .5;
		i++;
	}
}
void	matrix_print(Matrix *m)
{
	int	i;
	int	j;
	int	index;

	printf("Matrix (%d x%d):\n", m->rows,m->cols);

	i = 0;
	while (i < m->rows)
	{
		j = 0;
		while(j < m->cols)
		{
			index = (i * m->cols) + j;
			printf("%8.4f", m->data[index]);
			j++;
		}
		printf("\n");
		i++;
	}
	printf("\n");
}
void	matrix_add(Matrix  *m1, Matrix *m2)
{
	int	i;
	int	total;

	if(m1->rows != m2->rows ||m1->cols != m2->cols)
		return ;
	i = 0;
	total = m1->rows * m1->cols;
	while (i < total)
	{
		m1->data[i] += m2->data[i];
		i++;
	}
}
Matrix * matrix_sub(Matrix *m1, Matrix *m2)
{
	Matrix *result;
	int	i;
	int	total;

	if (m1->rows != m2->rows || m1->cols != m2->cols)
		return NULL;
	result = matrix_create(m1->rows, m1->cols);
	i = 0;
	total = m1->rows * m1->cols;
	while (i < total)
	{
		result->data[i] = m1->data[i] - m2->data[i];
		i++;
	}
	return result;
}
Matrix *matrix_dot(Matrix *m1,Matrix *m2)
{
	Matrix *result;
	int	i;
	int	j;
	int	k;
	double	sum;
	
	if (m1->cols != m2->cols)
		return NULL;
	result = matrix_create(m1->rows, m2->cols);
	i = 0;
	while (i < m1->rows)
	{
		j = 0;
		while (j < m2->cols)
		{
			sum = 0.0;
			k = 0;
			while (k < m1->cols)
			{
				sum +=m1->data[i*m1->cols + k] *m2->data[(k * m2->cols) + j];
				k++;
			}
			result->data[(i * result->cols) + k] = sum;
			j++;
		}
	}
	return result;
}
Matrix *matrix_transpose(Matrix *m)
{
	Matrix	*result;
	int	i;
	int	j;
	result = matrix_create(m->cols, m->rows);
	i = 0;
	while (i < m->rows)
	{
		j = 0;	
		while (j < m->cols)
		{
			result->data[(j * result ->cols) + i ]= m->data[(i * m->cols) + j];
			j++;
		}
		i++;
	}
	return result;
}
