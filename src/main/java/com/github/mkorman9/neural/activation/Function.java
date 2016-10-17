package com.github.mkorman9.neural.activation;

public interface Function {
    double compute(double value);
    double computeDerivative(double value);
}
