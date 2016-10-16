package com.github.mkorman9.neural.data;

public class HiddenLayerModel {
    private Matrix weights;
    private Vector bias;

    public HiddenLayerModel(int neurons, int dimension) {
        this.weights = Matrix.random(dimension, neurons);
        this.bias = Vector.random(neurons);
    }

    public HiddenLayerModel(Matrix weights, Vector bias) {
        this.weights = weights;
        this.bias = bias;
    }

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
