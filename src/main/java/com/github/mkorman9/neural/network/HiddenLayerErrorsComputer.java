package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;
import com.google.common.collect.Lists;

import java.util.List;

public class HiddenLayerErrorsComputer {
    private Model networkModel;
    private Function activationFunction;

    public HiddenLayerErrorsComputer(Model networkModel, Function activationFunction) {
        this.networkModel = networkModel;
        this.activationFunction = activationFunction;
    }

    public Vector compute(Vector hiddenLayerOutputs, Vector dv) {
        List<Double> vec = Lists.newArrayList();
        for (int i = 0; i < networkModel.getHiddenLayerNeuronsCount(); i++) {
            vec.add(activationFunction.compute(hiddenLayerOutputs.get(i)) * calculateError(dv, i));
        }
        return Vector.create(vec);
    }

    private double calculateError(Vector dv, int hiddenNeuronIndex) {
        double error = 0;
        for (int i = 0; i < dv.size(); i++) {
            error += networkModel.getOutputLayerModel().getWeights().value(hiddenNeuronIndex, i) * dv.get(i);
        }
        return error;
    }
}
