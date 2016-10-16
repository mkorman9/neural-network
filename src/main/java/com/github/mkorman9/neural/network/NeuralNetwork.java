package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.*;
import com.google.common.base.Preconditions;

public class NeuralNetwork {
    private int dimension;
    private int learningCyclesCount;
    private OutputLayerModel outputLayerModel;
    private HiddenLayerModel hiddenLayerModel;

    private HiddenLayerNeuronActivationComputer hiddenLayerNeuronActivationComputer;
    private OutputLayerNeuronActivationComputer outputLayerNeuronActivationComputer;
    private PredictionSuccessChecker predictionSuccessChecker;

    public NeuralNetwork(int dimension, Function activationFunction, int learningCyclesCount) {
        this.dimension = dimension;
        this.learningCyclesCount = learningCyclesCount;

        this.outputLayerModel = new OutputLayerModel(dimension);
        this.hiddenLayerModel = new HiddenLayerModel(dimension);

        this.hiddenLayerNeuronActivationComputer = new HiddenLayerNeuronActivationComputer(dimension, activationFunction);
        this.outputLayerNeuronActivationComputer = new OutputLayerNeuronActivationComputer(dimension, activationFunction);
        this.predictionSuccessChecker = new PredictionSuccessChecker();
    }

    public void learn(Matrix inputs, Vector outputs) {
        Preconditions.checkArgument(inputs.size() != 0, "Input vector cannot be empty");
        Preconditions.checkArgument(inputs.size() == outputs.size(), "Number of inputs should be equal to number of outputs");
        Preconditions.checkArgument(inputs.row(0).size() == dimension, "Number of input attributes should be equal to network dimension");

        int inputsCount = inputs.size();

        for (int iter = 0; iter < learningCyclesCount; iter++) {
            for (int i = 0; i < inputsCount; i++) {
                // find prediction
                Vector inputRow = inputs.row(i);
                Vector hiddenLayerOutputs = hiddenLayerNeuronActivationComputer.compute(inputRow, hiddenLayerModel);
                double outputLayerOutput = outputLayerNeuronActivationComputer.compute(hiddenLayerOutputs, outputLayerModel);

                // perform learning
                Learner learner = new Learner(hiddenLayerModel, outputLayerModel);
                learner.perform(inputRow, hiddenLayerOutputs, outputLayerOutput, outputs.get(i));
                this.hiddenLayerModel = learner.getHiddenLayerModel();
                this.outputLayerModel = learner.getOutputLayerModel();
            }
        }
    }

    public boolean predict(Vector input) {
        Vector hiddenLayerOutputs = hiddenLayerNeuronActivationComputer.compute(input, hiddenLayerModel);
        return predictionSuccessChecker.check(outputLayerNeuronActivationComputer.compute(hiddenLayerOutputs, outputLayerModel));
    }
}
