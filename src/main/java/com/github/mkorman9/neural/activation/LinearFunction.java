package com.github.mkorman9.neural.activation;

public class LinearFunction implements Function {
    @Override
    public double compute(double value) {
        return value;
    }

    @Override
    public double computeDerivative(double value) {
        return 1.0;
    }
}
