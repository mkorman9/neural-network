package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.HiddenLayerModel;
import com.github.mkorman9.neural.data.OutputLayerModel;
import com.github.mkorman9.neural.data.Vector;

class Learner {
    private static final double LEARNING_RATE = 0.05;

    private HiddenLayerModel hiddenLayerModel;
    private OutputLayerModel outputLayerModel;

    private OutputLayerWeightsComputer outputLayerWeightsComputer;
    private OutputLayerBiasComputer outputLayerBiasComputer;
    private HiddenLayerWeightsComputer hiddenLayerWeightsComputer;
    private HiddenLayerBiasComputer hiddenLayerBiasComputer;

    public Learner(HiddenLayerModel hiddenLayerModel, OutputLayerModel outputLayerModel) {
        this.hiddenLayerModel = hiddenLayerModel;
        this.outputLayerModel = outputLayerModel;

        this.outputLayerWeightsComputer = new OutputLayerWeightsComputer(outputLayerModel, LEARNING_RATE);
        this.outputLayerBiasComputer = new OutputLayerBiasComputer(outputLayerModel, LEARNING_RATE);
        this.hiddenLayerWeightsComputer = new HiddenLayerWeightsComputer(outputLayerModel, hiddenLayerModel, LEARNING_RATE);
        this.hiddenLayerBiasComputer = new HiddenLayerBiasComputer(outputLayerModel, hiddenLayerModel, LEARNING_RATE);
    }

    public void perform(Vector inputRow, Vector hiddenLayerOutputs, double outputLayerOutput, double expectedOutput) {
        double error = expectedOutput - outputLayerOutput;
        double dv = outputLayerOutput * (1 - outputLayerOutput) * error;

        outputLayerModel.setWeights(outputLayerWeightsComputer.compute(hiddenLayerOutputs, dv));
        outputLayerModel.setBias(outputLayerBiasComputer.compute(dv));
        hiddenLayerModel.setWeights(hiddenLayerWeightsComputer.compute(inputRow, hiddenLayerOutputs, dv));
        hiddenLayerModel.setBias(hiddenLayerBiasComputer.compute(hiddenLayerOutputs, dv));
    }

    public HiddenLayerModel getHiddenLayerModel() {
        return hiddenLayerModel;
    }

    public OutputLayerModel getOutputLayerModel() {
        return outputLayerModel;
    }
}
