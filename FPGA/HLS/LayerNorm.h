#pragma once
void LayerNorm(double input[512][768], double output_1[512][768], double fin_output[512][768],
double epsilon, double gamma[768], double beta[768]);