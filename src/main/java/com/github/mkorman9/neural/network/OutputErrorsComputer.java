package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

public class OutputErrorsComputer {
    private Model networkModel;

    public OutputErrorsComputer(Model networkModel) {
        this.networkModel = networkModel;
    }

    public Matrix compute(Vector hiddenLayerOutputs, Vector outputLayerOutputs, Vector expectedOutputs) {
        Matrix result = Matrix.zero(networkModel.getHiddenLayerNeuronsCount(), networkModel.getOutputLayerNeuronsCount());
        for (int i = 0; i < networkModel.getOutputLayerNeuronsCount(); i++) {
            for (int j = 0; j < networkModel.getHiddenLayerNeuronsCount(); j++) {
                double dv = computeValue(hiddenLayerOutputs.get(j), outputLayerOutputs.get(i), expectedOutputs.get(i));
                result.setValue(j, i, dv);
            }
        }
        return result;
    }

    private double computeValue(double hiddenLayerOutput, double outputLayerOutput, double expected) {
        return (expected - outputLayerOutput) * outputLayerOutput * (1 - outputLayerOutput) * hiddenLayerOutput;
    }
}
