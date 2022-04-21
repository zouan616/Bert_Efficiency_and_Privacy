#include <iostream>
#include"ffd.h"
#include <stdio.h>
#include <math.h>
#define PI acos(-1)
double GELU(double a)
{
    return 0.5*a*(1+ tanh(sqrt(2 / PI) * (a + 0.044715 * a * a * a)));
}
void FeedForward(double input[512][768],double W_1[768][768],double b_1, double b_2, double W_2[768][768], double output[512][768])
{
    double X_3[512][768];
    for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 768; ++j)
                for(int k = 0; k < 768; ++k)
                {
                    X_3[a][j] += GELU(input[a][k] * W_1[k][j] + b_1);
                }

    for(int a = 0; a < 512; ++a)
            for(int j = 0; j < 768; ++j)
                for(int k = 0; k < 768; ++k)
                {
                    output[a][j] += X_3[a][k] * W_2[k][j] + b_1;
                }
}