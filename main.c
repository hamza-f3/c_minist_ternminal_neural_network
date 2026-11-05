#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "nn.h"

#define MAX_LINE_LENGTH 10000

int	get_prediction (Matrix *m)
{
	int	i;
	int	max_index;
	double	max_val;

	i = 1;
	max_index = 0;
	max_val = m->data[0];
	while (i < m->rows * m->cols)
	{
		if(m->data[i] > max_val)
		{
			max_val = m->data[i];
			max_index = i;
		}
		i++;
	}
	return  max_index;
}
int main()
{
	FILE *file;
	char line[MAX_LINE_LENGTH];
	char *token;
	int label, prediction;
	int i;
	int	correct_predictions,total_samples;
	double learning_rate;
	NeuralNetwork *nn;
	Matrix *X, *Y;
	Cache *c;

	srand(time(NULL));
	nn = network_create(784, 128, 10);
	learning_rate = 0.1;

	file = fopen("mnist_train.csv", "r");
	if (!file)
	{
		printf("ERROR FILE");
		return 1;
	}
	X = matrix_create(784,1);
	Y = matrix_create(10, 1);

	correct_predictions = 0;
	total_samples = 0;
	printf("start training loop \n");
	
	while (fgets(line, MAX_LINE_LENGTH, file))
    {
        token = strtok(line, ",");
        if (!token) continue; // Skip completely empty lines safely
        
        label = atoi(token);

        // --- THE SAFETY NET ---
        if (label < 0 || label > 9)
        {
            printf("\n[Warning] Read a bad label '%d' at image %d. Skipping line...\n", label, total_samples);
            continue; // Skip this corrupted line and move to the next one!
        }
        // ----------------------

        i = 0;
        while (i < 10)
        {
            Y->data[i] = 0.0;
            i++;
        }
        Y->data[label] = 1.0;

        i = 0;
        while (i < 784)
        {
            token = strtok(NULL, ",");
            if (token)
                X->data[i] = (double)atoi(token) / 255.0;
            else
                X->data[i] = 0.0; // Fill missing pixels with black
            i++;
        }
        
        c = forward_pass(nn, X);
        prediction = get_prediction(c->A2);
        
        if(prediction == label)
        {
            correct_predictions++;
        }
        total_samples++;
        
        backward_pass(nn, c, X, Y, learning_rate);

        cache_free(c);
        
        if (total_samples % 1000 == 0)
        {
            printf("IMAGES PROCESSED: %d | Current Accuracy: %.2f%%\n", total_samples,
                    ((double)correct_predictions / total_samples) * 100.0);
        }
    }
	printf("\nTraining Complete! Final Accuracy: %.2f%%\n", 
           ((double)correct_predictions / total_samples) * 100.0);
	fclose(file);
	matrix_free(X);
    matrix_free(Y);
    network_free(nn);

    return 0;
}
