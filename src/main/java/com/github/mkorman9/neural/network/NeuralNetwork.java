package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.*;
import com.google.common.base.Preconditions;

public class NeuralNetwork {
    private int dimension;
    private int learningCyclesCount;
    private Model networkModel;

    private HiddenLayerNeuronActivationComputer hiddenLayerNeuronActivationComputer;
    private OutputLayerNeuronActivationComputer outputLayerNeuronActivationComputer;
    private PredictionSuccessChecker predictionSuccessChecker;

    public NeuralNetwork(Model networkModel, Function activationFunction) {
        this.networkModel = networkModel;
        this.dimension = networkModel.getHiddenLayerModel().getBias().size();
        this.learningCyclesCount = 0;

        this.hiddenLayerNeuronActivationComputer = new HiddenLayerNeuronActivationComputer(dimension, activationFunction);
        this.outputLayerNeuronActivationComputer = new OutputLayerNeuronActivationComputer(dimension, activationFunction);
        this.predictionSuccessChecker = new PredictionSuccessChecker();
    }

    public NeuralNetwork(int dimension, Function activationFunction, int learningCyclesCount) {
        this.dimension = dimension;
        this.learningCyclesCount = learningCyclesCount;

        this.networkModel = new Model(new HiddenLayerModel(dimension),
                                      new OutputLayerModel(dimension));

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
                Vector hiddenLayerOutputs = hiddenLayerNeuronActivationComputer.compute(inputRow, networkModel.getHiddenLayerModel());
                double outputLayerOutput = outputLayerNeuronActivationComputer.compute(hiddenLayerOutputs, networkModel.getOutputLayerModel());

                // perform learning
                Learner learner = new Learner(networkModel.getHiddenLayerModel(), networkModel.getOutputLayerModel());
                learner.perform(inputRow, hiddenLayerOutputs, outputLayerOutput, outputs.get(i));
                this.networkModel = new Model(learner.getHiddenLayerModel(),
                                              learner.getOutputLayerModel());
            }
        }
    }

    public boolean predict(Vector input) {
        Vector hiddenLayerOutputs = hiddenLayerNeuronActivationComputer.compute(input, networkModel.getHiddenLayerModel());
        return predictionSuccessChecker.check(outputLayerNeuronActivationComputer.compute(hiddenLayerOutputs, networkModel.getOutputLayerModel()));
    }

    public Model getModel() {
        return networkModel;
    }
}
