#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef struct
{
	Matrix	*W1;
	Matrix	*b1;
	Matrix	*W2;
	Matrix	*b2;
} NeuralNetwork;
typedef struct
{
	Matrix	*Z1;
	Matrix	*A1;
	Matrix	*Z2;
	Matrix	*A2;
} Cache;
NeuralNetwork * network_create(int	input_nodes,int hidden_nodes, int output_nodes);
void	network_free(NeuralNetwork *nn);
Cache	*forward_pass(NeuralNetwork *nn, Matrix *X);
void	backward_pass(NeuralNetwork *nn, Cache *c, Matrix *X, Matrix *Y, double learning_rate);
void	cache_free(Cache *c);
void apply_relu(Matrix *m);
Matrix *relu_prime(Matrix *m);
void apply_softmax(Matrix *m);
#endif
