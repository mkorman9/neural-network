package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.*;

class Learner {
    private static final double LEARNING_RATE = 0.05;

    private Model networkModel;

    private OutputLayerWeightsComputer outputLayerWeightsComputer;
    private OutputLayerBiasComputer outputLayerBiasComputer;
    private HiddenLayerWeightsComputer hiddenLayerWeightsComputer;
    private HiddenLayerBiasComputer hiddenLayerBiasComputer;
    private OutputErrorsComputer outputErrorsComputer;

    public Learner(Model networkModel) {
        this.networkModel = networkModel;

        this.outputLayerWeightsComputer = new OutputLayerWeightsComputer(networkModel, LEARNING_RATE);
        this.outputLayerBiasComputer = new OutputLayerBiasComputer(networkModel, LEARNING_RATE);
        this.hiddenLayerWeightsComputer = new HiddenLayerWeightsComputer(networkModel, LEARNING_RATE);
        this.hiddenLayerBiasComputer = new HiddenLayerBiasComputer(networkModel, LEARNING_RATE);
        this.outputErrorsComputer = new OutputErrorsComputer(networkModel);
    }

    public void perform(Vector inputRow, Vector hiddenLayerOutputs, Vector outputLayerOutput, Vector expectedOutput) {
        Matrix dv = outputErrorsComputer.compute(hiddenLayerOutputs, outputLayerOutput, expectedOutput);
        networkModel.getOutputLayerModel().setWeights(outputLayerWeightsComputer.compute(dv));
        networkModel.getOutputLayerModel().setBias(outputLayerBiasComputer.compute(dv));
        networkModel.getHiddenLayerModel().setWeights(hiddenLayerWeightsComputer.compute(inputRow, hiddenLayerOutputs,
                outputLayerOutput, expectedOutput));
        networkModel.getHiddenLayerModel().setBias(hiddenLayerBiasComputer.compute(hiddenLayerOutputs, dv));
    }
}
