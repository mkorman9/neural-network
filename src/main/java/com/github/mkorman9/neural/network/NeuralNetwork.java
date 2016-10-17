package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.activation.SigmoidFunction;
import com.github.mkorman9.neural.data.*;
import com.google.common.base.Preconditions;

public class NeuralNetwork {
    private Function activationFunction;
    private int learningCyclesCount;
    private Model networkModel;

    private HiddenLayerNeuronActivationComputer hiddenLayerNeuronActivationComputer;
    private OutputLayerNeuronActivationComputer outputLayerNeuronActivationComputer;

    private NeuralNetwork(Model networkModel) {
        this.activationFunction = new SigmoidFunction();
        this.networkModel = networkModel;
        this.learningCyclesCount = 0;

        this.hiddenLayerNeuronActivationComputer = new HiddenLayerNeuronActivationComputer(networkModel, activationFunction);
        this.outputLayerNeuronActivationComputer = new OutputLayerNeuronActivationComputer(networkModel, activationFunction);
    }

    private NeuralNetwork(int outputLayerNeurons, int hiddenLayerNeurons, int dimension, int learningCyclesCount) {
        this.activationFunction = new SigmoidFunction();
        this.networkModel = new Model(dimension,
                                        new HiddenLayerModel(hiddenLayerNeurons, dimension),
                                        new OutputLayerModel(outputLayerNeurons, hiddenLayerNeurons));
        this.learningCyclesCount = learningCyclesCount;

        this.hiddenLayerNeuronActivationComputer = new HiddenLayerNeuronActivationComputer(networkModel, activationFunction);
        this.outputLayerNeuronActivationComputer = new OutputLayerNeuronActivationComputer(networkModel, activationFunction);
    }

    public void learn(Matrix inputs, Matrix outputs) {
        Preconditions.checkArgument(inputs.size() != 0, "Input vector cannot be empty");
        Preconditions.checkArgument(inputs.size() == outputs.size(), "Number of inputs should be equal to number of outputs");
        Preconditions.checkArgument(inputs.row(0).size() == networkModel.getInputsCount(),
                "Number of input attributes should be equal to network dimension");

        int inputsCount = inputs.size();

        for (int iter = 0; iter < learningCyclesCount; iter++) {
            for (int i = 0; i < inputsCount; i++) {
                // find prediction
                Vector inputRow = inputs.row(i);
                Vector hiddenLayerOutputs = hiddenLayerNeuronActivationComputer.compute(inputRow);
                Vector outputLayerOutputs = outputLayerNeuronActivationComputer.compute(hiddenLayerOutputs);

                // perform learning
                Learner learner = new Learner(networkModel, activationFunction);
                learner.perform(inputRow, hiddenLayerOutputs, outputLayerOutputs, outputs.row(i));
            }
        }
    }

    public Vector predict(Vector input) {
        Vector hiddenLayerOutputs = hiddenLayerNeuronActivationComputer.compute(input);
        return outputLayerNeuronActivationComputer.compute(hiddenLayerOutputs);
    }

    public Model getModel() {
        return networkModel;
    }

    public static Builder build() {
        return new Builder();
    }

    public static NeuralNetwork buildFromModel(Model networkModel) {
        return new NeuralNetwork(networkModel);
    }

    public static class Builder {
        private int outputLayerNeurons;
        private int hiddenLayerNeurons;
        private int dimension;
        private int learningCyclesCount;

        public Builder outputLayerNeurons(int outputLayerNeurons) {
            this.outputLayerNeurons = outputLayerNeurons;
            return this;
        }

        public Builder hiddenLayerNeurons(int hiddenLayerNeurons) {
            this.hiddenLayerNeurons = hiddenLayerNeurons;
            return this;
        }

        public Builder dimension(int dimension) {
            this.dimension = dimension;
            return this;
        }

        public Builder learningCyclesCount(int learningCyclesCount) {
            this.learningCyclesCount = learningCyclesCount;
            return this;
        }

        public NeuralNetwork done() {
            return new NeuralNetwork(outputLayerNeurons, hiddenLayerNeurons, dimension, learningCyclesCount);
        }
    }
}
