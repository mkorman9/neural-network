package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class HiddenLayerBiasComputer {
    private Model networkModel;
    private double learningRate;

    public HiddenLayerBiasComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Vector compute(Vector hiddenLayerOutputs, Matrix dv) {
        return networkModel.getHiddenLayerModel().getBias();
    }
}
