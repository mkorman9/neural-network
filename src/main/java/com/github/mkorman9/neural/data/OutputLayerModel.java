package com.github.mkorman9.neural.data;

public class OutputLayerModel {
    private Vector weights;
    private double bias;

    public OutputLayerModel(int neurons) {
        this.weights = Vector.random(neurons);
        this.bias = RandomValue.generate();
    }

    public OutputLayerModel(Vector weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public Vector getWeights() {
        return weights;
    }

    public void setWeights(Vector weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
