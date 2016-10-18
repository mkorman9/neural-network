package com.github.mkorman9.neural;

import com.github.mkorman9.neural.activation.SigmoidFunction;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;
import com.github.mkorman9.neural.data.parser.CsvReader;
import com.github.mkorman9.neural.network.NeuralNetwork;
import com.github.mkorman9.neural.network.reader.DefaultReader;
import com.github.mkorman9.neural.network.reader.Reader;
import com.github.mkorman9.neural.network.writer.DefaultWriter;
import com.github.mkorman9.neural.network.writer.Writer;

import java.io.File;

public class MultilabelTest {
    public static void main(String[] args) {
        File inputFile = new File(args[1]);
        File outputFile = new File(args[2]);

        if (args[0].equals("train")) {
            train(inputFile, outputFile);
        }
        else if (args[0].equals("predict")) {
            predict(inputFile, outputFile);
        }
    }

    private static void train(File inputFile, File outputFile) {
        Matrix input = new CsvReader().readFromFile(inputFile);
        Matrix output = new CsvReader().readFromFile(outputFile);

        NeuralNetwork neuralNetwork = NeuralNetwork.buildNew()
                                                    .outputLayerNeurons(3)
                                                    .hiddenLayerNeurons(5)
                                                    .inputLayerNeurons(5)
                                                    .learningCyclesCount(2000)
                                                    .activationFunction(new SigmoidFunction())
                                                    .done();
        neuralNetwork.learn(input, output);

        Writer writer = new DefaultWriter();
        writer.write(neuralNetwork.getModel(), new File("target/multilabel_model.txt"));
    }

    private static void predict(File inputFile, File outputFile) {
        Matrix input = new CsvReader().readFromFile(inputFile);
        Matrix output = new CsvReader().readFromFile(outputFile);

        Reader reader = new DefaultReader();
        Model model = reader.read(new File("target/multilabel_model.txt"));

        NeuralNetwork neuralNetwork = NeuralNetwork.buildFromModel()
                .model(model)
                .activationFunction(new SigmoidFunction())
                .done();

        int success = 0;
        for (int i = 0; i < input.size(); i++) {
            Vector prediction = neuralNetwork.predict(input.row(i));
            int partialSuccesses = 0;
            for (int j = 0; j < output.row(i).size(); j++) {
                if ((prediction.get(j) >= 0.5 && output.row(i).get(j) == 1.0) || (prediction.get(j) < 0.5 && output.row(i).get(j) == 0.0)) {
                    partialSuccesses++;
                }
            }

            if (partialSuccesses == output.row(i).size()) {
                success++;
            }
        }

        System.out.printf("Guessed %d/%d (%f%%)\n", success, input.size(), (double) success / (double) input.size() * 100.0);
    }
}
