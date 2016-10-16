package com.github.mkorman9.neural.data;

public class HiddenLayerModel {
    private Vector bias;
    private Matrix weights;

    public HiddenLayerModel(int dimension) {
        this.bias = Vector.random(dimension);
        this.weights = Matrix.random(dimension, dimension);
    }

    public HiddenLayerModel(Vector bias, Matrix weights) {
        this.bias = bias;
        this.weights = weights;
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
