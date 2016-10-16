package com.github.mkorman9.neural;

import com.github.mkorman9.neural.activation.SigmoidFunction;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;
import com.github.mkorman9.neural.data.interpreter.SingleClassOutputsInterpreter;
import com.github.mkorman9.neural.data.parser.CsvReader;
import com.github.mkorman9.neural.network.NeuralNetwork;
import com.github.mkorman9.neural.network.reader.DefaultReader;
import com.github.mkorman9.neural.network.reader.Reader;
import com.github.mkorman9.neural.network.writer.DefaultWriter;
import com.github.mkorman9.neural.network.writer.Writer;

import java.io.File;

public class SimpleTest {
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
        Vector output = new SingleClassOutputsInterpreter().interpret(new CsvReader().readFromFile(outputFile));

        NeuralNetwork neuralNetwork = new NeuralNetwork(3, 2, new SigmoidFunction(), 1000);
        neuralNetwork.learn(input, output);

        Writer writer = new DefaultWriter();
        writer.write(neuralNetwork.getModel(), new File("target/simple_model.txt"));
    }

    private static void predict(File inputFile, File outputFile) {
        Matrix input = new CsvReader().readFromFile(inputFile);
        Vector output = new SingleClassOutputsInterpreter().interpret(new CsvReader().readFromFile(outputFile));

        Reader reader = new DefaultReader();
        Model model = reader.read(new File("target/simple_model.txt"));

        NeuralNetwork neuralNetwork = new NeuralNetwork(model, new SigmoidFunction());

        int success = 0;
        for (int i = 0; i < input.size(); i++) {
            Vector inputRow = input.row(i);
            boolean prediction = neuralNetwork.predictTrueFalse(inputRow);

            if ((prediction && output.get(i) == 1.0) || (!prediction && output.get(i) == 0.0)) {
                success += 1;
            }
        }

        System.out.printf("Guessed %d/%d (%f%%)\n", success, input.size(), (double) success / (double) input.size() * 100.0);
    }
}
