package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.*;
import com.google.common.base.Preconditions;

public class NeuralNetwork {
    private Function activationFunction;
    private int learningCyclesCount;
    private Model networkModel;

    private HiddenLayerNeuronActivationComputer hiddenLayerNeuronActivationComputer;
    private OutputLayerNeuronActivationComputer outputLayerNeuronActivationComputer;

    private NeuralNetwork(Model networkModel, Function activationFunction) {
        this.activationFunction = activationFunction;
        this.networkModel = networkModel;
        this.learningCyclesCount = 0;

        this.hiddenLayerNeuronActivationComputer = new HiddenLayerNeuronActivationComputer(networkModel, activationFunction);
        this.outputLayerNeuronActivationComputer = new OutputLayerNeuronActivationComputer(networkModel, activationFunction);
    }

    private NeuralNetwork(int outputLayerNeurons, int hiddenLayerNeurons, int inputLayerNeurons,
                          int learningCyclesCount, Function activationFunction) {
        this.activationFunction = activationFunction;
        this.networkModel = new Model(inputLayerNeurons,
                                        new HiddenLayerModel(hiddenLayerNeurons, inputLayerNeurons),
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

    public static FromPropertiesBuilder buildNew() {
        return new FromPropertiesBuilder();
    }

    public static FromModelBuilder buildFromModel() {
        return new FromModelBuilder();
    }

    public static class FromModelBuilder {
        private Model model;
        private Function activationFunction;

        public FromModelBuilder model(Model model) {
            this.model = model;
            return this;
        }

        public FromModelBuilder activationFunction(Function activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public NeuralNetwork done() {
            return new NeuralNetwork(model, activationFunction);
        }
    }

    public static class FromPropertiesBuilder {
        private int outputLayerNeurons;
        private int hiddenLayerNeurons;
        private int inputLayerNeurons;
        private int learningCyclesCount;
        private Function activationFunction;

        public FromPropertiesBuilder outputLayerNeurons(int outputLayerNeurons) {
            this.outputLayerNeurons = outputLayerNeurons;
            return this;
        }

        public FromPropertiesBuilder hiddenLayerNeurons(int hiddenLayerNeurons) {
            this.hiddenLayerNeurons = hiddenLayerNeurons;
            return this;
        }

        public FromPropertiesBuilder inputLayerNeurons(int inputLayerNeurons) {
            this.inputLayerNeurons = inputLayerNeurons;
            return this;
        }

        public FromPropertiesBuilder learningCyclesCount(int learningCyclesCount) {
            this.learningCyclesCount = learningCyclesCount;
            return this;
        }

        public FromPropertiesBuilder activationFunction(Function activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public NeuralNetwork done() {
            return new NeuralNetwork(outputLayerNeurons, hiddenLayerNeurons, inputLayerNeurons,
                    learningCyclesCount, activationFunction);
        }
    }
}
