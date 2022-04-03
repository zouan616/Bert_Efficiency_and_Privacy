#include <iostream>
using namespace std;
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fstream>

double input[512][768];
double W_Q[768][64][12];
double W_K[768][64][12];
double W_V[768][64][12];
double output[512][768];

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

void top_func(double input[512][768], double W_Q[768][64][12], double W_K[768][64][12], double W_V[768][64][12], double output[512][768])
{
    ofstream pre("pre.txt");
    for (int i = 0; i < 12; ++i)
    {
        double Q[512][64]{0}, K[512][64]{0}, V[512][64]{0}, k_transpose[64][512]{0}, half[512][512]{0};
        double pre_softmax[512][64]{0};
        for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 64; ++j)
                for(int k = 0; k < 768; ++k)
                {
                    Q[a][j] += input[a][k] * W_Q[k][j][i];
                    K[a][j] += input[a][k] * W_K[k][j][i];
                    V[a][j] += input[a][k] * W_V[k][j][i];
                }
        for (int a = 0; a < 512; ++a)//computing transpose of K
        {
            for (int j = 0; j < 64; ++j) 
                {
                k_transpose[j][a] = K[a][j];
                }
        }
        for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 512; ++j)
                for(int k = 0; k < 64; ++k)
                {
                    half[a][j] += Q[a][k] * k_transpose[k][j] / 8.0;
                }
        for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 64; ++j)
                for(int k = 0; k < 512; ++k)
                {
                    pre_softmax[a][j] += half[a][k] * V[k][j];
                }

        for (int a = 0; a < 512; a++)
        {
            for (int b = 0; b < 64; b++)
            {
                pre << pre_softmax[a][b] << "  ";
            }
            pre << endl;
        }

        for (int a = 0; a < 512; ++a)
        {
            softmax(pre_softmax[a], 64);
        }
        for (int a = 0; a < 512; ++a)
        {
            for (int j = 64 * i; j < 64 * (i + 1); j++)
            {
                output[a][j] = pre_softmax[a][j - 64 * i];
            } 
        } 
    }  
    pre.close(); 
}

int main()
{
    ofstream in("in2.txt");
    for (int i = 0; i < 512; i++)
    {
        for (int j = 0; j < 768; j++)
        {
            input[i][j] = .7 + j / 350.0;
            in << input[i][j] << " ";
        }
        in << endl;
    }
    for (int i = 0; i < 768; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            for (int k = 0; k < 12; k++)
            {
                W_Q[i][j][k] = .3  + k / 6.0;
                W_K[i][j][k] = .5 + i / 340 + k / 7.0;
                W_V[i][j][k] = .6 + i / 330  + k / 8.0;
            }
        }
    }
    ofstream out("out2.txt");
    top_func(input, W_Q, W_K, W_V, output);
    for (int i = 0; i < 512; i++)
    {
        for (int j = 0; j < 768; j++)
        {
            out << output[i][j] <<" ";
        }
        out << endl;
    }
    out.close();
    in.close();
    return 0;
}