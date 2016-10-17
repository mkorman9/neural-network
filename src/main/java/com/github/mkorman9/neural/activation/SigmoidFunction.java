package com.github.mkorman9.neural.activation;

public class SigmoidFunction implements Function {
    @Override
    public double compute(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    @Override
    public double computeDerivative(double value) {
        return value * (1.0 - value);
    }
}
