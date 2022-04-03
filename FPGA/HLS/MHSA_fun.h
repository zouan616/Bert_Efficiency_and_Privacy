#pragma once
void softmax(double* input, int size);
void top_func(double input[512][768], const double W_Q[768][64][12], const double W_K[768][64][12], const double W_V[768][64][12], double output [512][768]);