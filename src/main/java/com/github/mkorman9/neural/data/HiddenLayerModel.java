package com.github.mkorman9.neural.data;

public class HiddenLayerModel extends LayerModel {
    public HiddenLayerModel(int neurons, int inputsCount) {
        this.weights = Matrix.random(inputsCount, neurons);
        this.bias = Vector.random(neurons);
    }

    public HiddenLayerModel(Matrix weights, Vector bias) {
        this.weights = weights;
        this.bias = bias;
    }
}
