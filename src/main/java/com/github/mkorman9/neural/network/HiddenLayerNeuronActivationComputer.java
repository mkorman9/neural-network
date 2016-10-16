package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.HiddenLayerModel;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;

class HiddenLayerNeuronActivationComputer {
    private int dimension;
    private Function activationFunction;

    public HiddenLayerNeuronActivationComputer(int dimension, Function activationFunction) {
        this.dimension = dimension;
        this.activationFunction = activationFunction;
    }

    public Vector compute(Vector inputRow, HiddenLayerModel hiddenLayerModel) {
        Vector neuronOutputs = Vector.zero(dimension);
        for (int i = 0; i < dimension; i++) {
            double sum = computeWeightSum(inputRow, hiddenLayerModel.getWeights(), i);
            double activation = activationFunction.compute(sum + hiddenLayerModel.getBias().get(i));
            neuronOutputs.set(i, activation);
        }
        return neuronOutputs;
    }

    private double computeWeightSum(Vector inputRow, Matrix hiddenLayerWeights, int i) {
        double sum = 0;
        for (int j = 0; j < dimension; j++) {
            sum += inputRow.get(j) * hiddenLayerWeights.value(j, i);
        }
        return sum;
    }
}
