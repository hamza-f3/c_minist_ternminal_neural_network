#include "nn.h"
#include "matrix.h"
#include <stdlib.h>
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
Cache	*forward_pass(NeuralNetwork *nn, Matrix *X)
{
	Cache	*c;

	c = (Cache *)malloc(sizeof(Cache));

	// Z1 = W1 * X + b1;
	
	c->Z1 = matrix_dot(nn->W1,X);
	matrix_add(c->Z1, nn->b1);

	//A1 = Relu(Z1)
	
	c->A1 = matrix_copy(c->Z1);
	apply_relu(c->A1);

	//Z2 = W2 * A1 + b2;
	
	c->Z2 = matrix_dot(nn->W2 ,c->A1);
	matrix_add(c->Z2, nn->b2);

	//A2 = softmax(Z2)
	c->A2 = matrix_copy(c->Z2);
	apply_softmax(c->A2);
	
	return c;
}
void	cache_free(Cache *c)
{
	if(c)
	{
		matrix_free(c->Z1);
		matrix_free(c->A1);
		matrix_free(c->Z2);
		matrix_free(c->A2);
		free(c);
	}
}

//	backpropagation and Weight Update
void backward_pass(NeuralNetwork *nn, Cache *c, Matrix *X, Matrix *Y, double learning_rate)
{
	Matrix	*dZ2, *dW2, *db2;
	Matrix	*dZ1, *dW1, *db1;
	Matrix	*A1_T, *W2_T, *X_T;
	Matrix	*dZ1_temp, *rp_Z1;

	// -- Output Layer Gradients --
	// dZ2 = A2 - Y
	dZ2 = matrix_sub(c->A2, Y);

	//dW2 = dZ2 * A1^T
	A1_T = matrix_transpose(c->A1);
	dW2 = matrix_dot(dZ2, A1_T);
	matrix_free(A1_T);
	
	//db2 is simply dZ2 for a single sample
	db2 = matrix_copy(dZ2);

	//-- Hidden Layer Gradiants --
	// dZ1 = (W2^T * dZ2) . * relu_prime(Z1)
	W2_T = matrix_transpose(nn->W2);
	dZ1_temp = matrix_dot(W2_T, dZ2);
	matrix_free(W2_T);

	rp_Z1 = relu_prime(c->Z1);
	dZ1 = matrix_multiply_elementwise(dZ1_temp, rp_Z1);
	matrix_free(dZ1_temp);
	matrix_free(rp_Z1);

	//dW1 = dZ1 * X^T
	X_T = matrix_transpose(X);
	dW1 = matrix_dot(dZ1, X_T);
	matrix_free(X_T);

	//db1 is simply dZ1 for single sample
	db1 = matrix_copy(dZ1);

	// -- Update Weights and biases (Gradiants Descent)
	// W = W - (learning_rate * dW)
	
	//Scale gradients by negative learning_rate
	matrix_scale(dW1, -learning_rate);
	matrix_scale(db1, -learning_rate);
	matrix_scale(dW2, -learning_rate);
	matrix_scale(db2, -learning_rate);

	// Add the negative gradients to update the weights in-place
	matrix_add(nn->W1, dW1);
	matrix_add(nn->b1, db1);
	matrix_add(nn->W2, dW2);
	matrix_add(nn->b2, db1);

	// --cleanup matrices --
	matrix_free(dZ2);
	matrix_free(dW2);
	matrix_free(db2);
	matrix_free(dZ1);
	matrix_free(dW2);
	matrix_free(db2);
}
