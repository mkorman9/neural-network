package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class OutputLayerBiasComputer {
    private Model networkModel;
    private double learningRate;

    public OutputLayerBiasComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Vector compute(Matrix dv) {
        return networkModel.getOutputLayerModel().getBias();
    }
}
