package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.HiddenLayerModel;
import com.github.mkorman9.neural.data.OutputLayerModel;
import com.github.mkorman9.neural.data.Vector;

class HiddenLayerBiasComputer {
    private OutputLayerModel outputLayerModel;
    private HiddenLayerModel hiddenLayerModel;
    private double learningRate;

    public HiddenLayerBiasComputer(OutputLayerModel outputLayerModel, HiddenLayerModel hiddenLayerModel, double learningRate) {
        this.outputLayerModel = outputLayerModel;
        this.hiddenLayerModel = hiddenLayerModel;
        this.learningRate = learningRate;
    }

    public Vector compute(Vector hiddenLayerOutputs, double dv) {
        Vector newBias = Vector.zero(hiddenLayerOutputs.size());
        for (int i = 0; i < hiddenLayerOutputs.size(); i++) {
            double dbi = hiddenLayerOutputs.get(i) * (1 - hiddenLayerOutputs.get(i)) * outputLayerModel.getWeights().get(i) * dv;
            double db = learningRate * dbi;
            newBias.set(i, hiddenLayerModel.getBias().get(i) + db);
        }
        return newBias;
    }
}
