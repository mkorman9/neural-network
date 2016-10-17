package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

public class OutputErrorsComputer {
    private Model networkModel;
    private Function activationFunction;

    public OutputErrorsComputer(Model networkModel, Function activationFunction) {
        this.networkModel = networkModel;
        this.activationFunction = activationFunction;
    }

    public Vector compute(Vector outputLayerOutputs, Vector expectedOutputs) {
        Vector result = Vector.zero(networkModel.getOutputLayerNeuronsCount());
        for (int i = 0; i < networkModel.getOutputLayerNeuronsCount(); i++) {
            double dv = computeValue(outputLayerOutputs.get(i), expectedOutputs.get(i));
            result.set(i, dv);
        }
        return result;
    }

    private double computeValue(double outputLayerOutput, double expected) {
        return (expected - outputLayerOutput) * activationFunction.computeDerivative(outputLayerOutput);
    }
}
