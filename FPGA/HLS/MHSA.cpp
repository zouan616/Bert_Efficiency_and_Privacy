#include <iostream>
#include"func.h"
#include <stdio.h>
#include <math.h>
void softmax(double* input, int size)
{
	int i;
	double m, sum, constant;
	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}
	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - m);
	}
	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - constant);
	}
}

void top_func(double input[512][768], const double W_Q[768][64][12], const double W_K[768][64][12], const double W_V[768][64][12], double output [512][768])
{
    for (int i = 0; i < 12; i++)
    {
        double Q[512][64], K[512][64], V[512][64], k_transpose[64][512], half[512][512], pre_softmax[512][64];
        for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 64; ++j)
                for(int k = 0; k < 768; ++k)
                {
                    Q[a][j] += input[a][k] * W_Q[k][j][i];
                    K[a][j] += input[a][k] * W_K[k][j][i];
                    V[a][j] += input[a][k] * W_V[k][j][i];
                }
        for (int a = 0; a < 512; ++a)//computing transpose of K
            for (int j = 0; j < 64; ++j) {
                k_transpose[j][a] = K[a][j];
            }
        for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 512; ++j)
                for(int k = 0; k < 64; ++k)
                {
                    half[a][j] += (Q[a][k] * k_transpose[k][j]) / 8;
                }
        for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 64; ++j)
                for(int k = 0; k < 512; ++k)
                {
                    pre_softmax[a][j] += half[a][k] * V[k][j];
                }
        for (int a = 0; a < 512; a++)
        {
            softmax(pre_softmax[a], 64);
        }
        for (int a = 0; a < 512; a++)
        {
            for (int j = 64*(i); j < 64*(1+i); j++)
            {
                output[a][j] = pre_softmax[a][j - 64*i];
            } 
        } 
    }   
}
