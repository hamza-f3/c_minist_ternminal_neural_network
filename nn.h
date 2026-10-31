#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef struct
{
	Matrix	*W1;
	Matrix	*b1;
	Matrix	*W2;
	Matrix	*b2;
}NeuralNetwork;
NeuralNetwork * network_create(int	input_nodes,int hidden_nodes, int output_nodes);
void	netwoek_free(NeuralNetwork *nn);

void apply_relu(Matrix *m);
Matrix *relu_prime(Matrix *m);
void apply_softmax(Matrix *m);
#endif
