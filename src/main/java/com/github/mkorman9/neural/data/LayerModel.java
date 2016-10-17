package com.github.mkorman9.neural.data;

public class LayerModel {
    protected Matrix weights;
    protected Vector bias;

    public Vector getBias() {
        return bias;
    }

    public void setBias(Vector bias) {
        this.bias = bias;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }
}
