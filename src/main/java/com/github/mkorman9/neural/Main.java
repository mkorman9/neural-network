package com.github.mkorman9.neural;

import com.github.mkorman9.neural.activation.SigmoidFunction;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;
import com.github.mkorman9.neural.data.interpreter.MultiClassOutputsInterpreter;
import com.github.mkorman9.neural.data.parser.CsvReader;
import com.github.mkorman9.neural.network.NeuralNetwork;
import com.github.mkorman9.neural.network.writer.DefaultWriter;
import com.github.mkorman9.neural.network.writer.Writer;
import com.google.common.collect.Lists;

import java.io.File;

public class Main {
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
        Matrix output = new MultiClassOutputsInterpreter().interpret(new CsvReader().readFromFile(outputFile),
                Lists.newArrayList(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));

        for (int i = 0; i < output.size(); i++) {
            Vector labels = output.row(0);
            NeuralNetwork neuralNetwork = new NeuralNetwork(input.row(0).size(), new SigmoidFunction(), 100);
            neuralNetwork.learn(input, labels);

            Writer writer = new DefaultWriter();
            writer.write(neuralNetwork.getModel(), new File(String.format("target/model_%d.txt", i)));

            System.out.printf("Wrote model for '%d' to target/model_%d.txt\n", i, i);
        }
    }

    private static void predict(File inputFile, File outputFile) {

    }
}
