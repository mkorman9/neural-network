package com.github.mkorman9.neural.data;

public class OutputLayerModel extends LayerModel {
    public OutputLayerModel(int neurons, int hiddenLayerOutputs) {
        this.weights = Matrix.random(hiddenLayerOutputs, neurons);
        this.bias = Vector.random(neurons);
    }

    public OutputLayerModel(Matrix weights, Vector bias) {
        this.weights = weights;
        this.bias = bias;
    }
}
