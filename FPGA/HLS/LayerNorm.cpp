#include <iostream>
#include"LayerNorm.h"
#include <stdio.h>
#include <math.h>

void LayerNorm(double input[512][768], double output_1[512][768], double fin_output[512][768],
double epsilon, double gamma[768], double beta[768])
{
    for (int i = 0; i < 512; i++)
    {
        double mean, SD;
        double sum = 0;
        for (int j = 0; j < 768; j++)
        {
            sum += input[i][j] + output_1[i][j];
        }
        mean = sum / 768;
        for (int j = 0; j < 768; j++)
        {
            SD += (input[i][j] + output_1[i][j] - mean) * (input[i][j] + output_1[i][j] - mean);
        }
        SD = SD / 768;

        double half[768];
        for (int j = 0; j < 768; j++)
        {
            half[j] = (input[i][j] + output_1[i][j] - mean) / sqrt(SD + epsilon);
            fin_output[i][j] = half[j] * gamma[j] + beta[j];
        }
    }
}