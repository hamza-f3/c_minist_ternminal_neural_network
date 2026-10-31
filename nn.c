#include "nn.h"
#include "matrix.h"
NeuralNetwork *network_create(int input_nodes, int hidden_nodes, int output_nodes)
{
	NeuralNetwork *nn;
	nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
	if(!nn)
		return NULL;
	nn->W1 = matrix_create(hidden_nodes, input_nodes);
	nn->b1 = matrix_create(hidden_nodes, 1);

	nn->W2 = matrix_create(output_nodes, hidden_nodes);
	nn->b2 = matrix_create(output_nodes, 1);

	matrix_randomize(nn->W1);
	matrix_randomize(nn->W2);
	
	return nn;
}
void network_free(NeuralNetwork *nn)
{
	if (nn)
	{
		matrix_free (nn->W1);
		matrix_free (nn->b1);
		matrix_free (nn->W2);
		matrix_free (nn->b2);
		free(nn);
	}
}
void apply_relu(Matrix *m)
{
	int	i;
	int	total;

	i = 0;
	total = m->rows * m->cols;
	while (i < total)
	{
		if(m->data[i] < 0.0)
			m->data[i] = 0.0;
		i++;
	}
}
Matrix * relu_prime(Matrix *m)
{
	Matrix	*result;
	int	i;
	int	total;

	result = matrix_create(m->rows, m->cols);
	i = 0;
	total = m->rows * m->cols;
	while (i < total)
	{
		if(m->data[i] > 0.0)
			result->data[i] = 1.0;
		else
			result->data[i] = 0.0;
		i++;
	}
	return result;
}
void	apply_softmax(Matrix *m)
{
	int	i;
	int	total;
	double	max_value;
	double	sum;

	total = m->rows * m->cols;
	max_value = m->data[0];
	i = 1;
	while (i < total)
	{
		if(m->data[i] > max_value)
			max_value = m->data[i];
		i++;
	}
	sum = 0.0;
	i = 0;
	while (i < total)
	{
		m->data[i] = exp(m->data[i] - max_value);
		sum +=m->data[i];
		i++;
	}
	i = 0;
	while(i < total)
	{
		m->data[i] /= sum;
		i++;
	}
}
