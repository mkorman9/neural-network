package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class HiddenLayerNeuronActivationComputer {
    private Model networkModel;
    private Function activationFunction;

    public HiddenLayerNeuronActivationComputer(Model networkModel, Function activationFunction) {
        this.networkModel = networkModel;
        this.activationFunction = activationFunction;
    }

    public Vector compute(Vector inputRow) {
        Vector neuronOutputs = Vector.zero(networkModel.getHiddenLayerNeuronsCount());
        for (int i = 0; i < networkModel.getHiddenLayerNeuronsCount(); i++) {
            double sum = computeWeightSum(inputRow, networkModel.getHiddenLayerModel().getWeights(), i);
            double activation = activationFunction.compute(sum + networkModel.getHiddenLayerModel().getBias().get(i));
            neuronOutputs.set(i, activation);
        }
        return neuronOutputs;
    }

    private double computeWeightSum(Vector inputRow, Matrix hiddenLayerWeights, int i) {
        double sum = 0;
        for (int j = 0; j < networkModel.getInputsCount(); j++) {
            sum += inputRow.get(j) * hiddenLayerWeights.value(j, i);
        }
        return sum;
    }
}
