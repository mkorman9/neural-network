package com.github.mkorman9.neural.activation;

public class SigmoidFunction implements Function {
    @Override
    public Double compute(Double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }
}
