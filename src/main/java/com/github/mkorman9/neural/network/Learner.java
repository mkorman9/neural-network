package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.*;

class Learner {
    private static final double LEARNING_RATE = 0.05;

    private Function activationFunction;
    private Model networkModel;

    private OutputLayerWeightsComputer outputLayerWeightsComputer;
    private OutputLayerBiasComputer outputLayerBiasComputer;
    private HiddenLayerWeightsComputer hiddenLayerWeightsComputer;
    private HiddenLayerBiasComputer hiddenLayerBiasComputer;
    private OutputErrorsComputer outputErrorsComputer;
    private HiddenLayerErrorsComputer hiddenLayerErrorsComputer;

    public Learner(Model networkModel, Function activationFunction) {
        this.networkModel = networkModel;

        this.outputLayerWeightsComputer = new OutputLayerWeightsComputer(networkModel, LEARNING_RATE);
        this.outputLayerBiasComputer = new OutputLayerBiasComputer(networkModel, LEARNING_RATE);
        this.hiddenLayerWeightsComputer = new HiddenLayerWeightsComputer(networkModel, LEARNING_RATE);
        this.hiddenLayerBiasComputer = new HiddenLayerBiasComputer(networkModel, LEARNING_RATE);
        this.outputErrorsComputer = new OutputErrorsComputer(networkModel, activationFunction);
        this.hiddenLayerErrorsComputer = new HiddenLayerErrorsComputer(networkModel, activationFunction);
        this.activationFunction = activationFunction;
    }

    public void perform(Vector inputRow, Vector hiddenLayerOutputs, Vector outputLayerOutput, Vector expectedOutput) {
        Vector dv = outputErrorsComputer.compute(outputLayerOutput, expectedOutput);
        Vector dw = hiddenLayerErrorsComputer.compute(hiddenLayerOutputs, dv);
        networkModel.getOutputLayerModel().setWeights(outputLayerWeightsComputer.compute(hiddenLayerOutputs, dv));
        networkModel.getOutputLayerModel().setBias(outputLayerBiasComputer.compute(dv));
        networkModel.getHiddenLayerModel().setWeights(hiddenLayerWeightsComputer.compute(inputRow, dw));
        networkModel.getHiddenLayerModel().setBias(hiddenLayerBiasComputer.compute(hiddenLayerOutputs, dv));
    }
}
