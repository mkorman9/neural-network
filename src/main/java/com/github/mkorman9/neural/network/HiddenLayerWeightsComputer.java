package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.HiddenLayerModel;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.OutputLayerModel;
import com.github.mkorman9.neural.data.Vector;

class HiddenLayerWeightsComputer {
    private OutputLayerModel outputLayerModel;
    private HiddenLayerModel hiddenLayerModel;
    private double learningRate;

    public HiddenLayerWeightsComputer(OutputLayerModel outputLayerModel, HiddenLayerModel hiddenLayerModel, double learningRate) {
        this.outputLayerModel = outputLayerModel;
        this.hiddenLayerModel = hiddenLayerModel;
        this.learningRate = learningRate;
    }

    public Matrix compute(Vector inputRow, Vector hiddenLayerOutputs, double dv) {
        Vector dwi = Vector.zero(hiddenLayerOutputs.size());
        Matrix dw = Matrix.zero(hiddenLayerOutputs.size(), hiddenLayerOutputs.size());
        Matrix newHiddenLayerWeights = Matrix.zero(hiddenLayerOutputs.size(), hiddenLayerOutputs.size());

        for (int i = 0; i < hiddenLayerOutputs.size(); i++) {
            double value = hiddenLayerOutputs.get(i) * (1 - hiddenLayerOutputs.get(i)) * outputLayerModel.getWeights().get(i) * dv;
            dwi.set(i, value);

            for (int j = 0; j < hiddenLayerOutputs.size(); j++) {
                dw.setValue(j, i, learningRate * dwi.get(i) * inputRow.get(j));
                newHiddenLayerWeights.setValue(j, i, hiddenLayerModel.getWeights().value(j, i) + dw.value(j, i));
            }
        }

        return newHiddenLayerWeights;
    }
}
