package com.github.mkorman9.neural;

import com.github.mkorman9.neural.activation.SigmoidFunction;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;
import com.github.mkorman9.neural.data.interpreter.MultiClassOutputsInterpreter;
import com.github.mkorman9.neural.data.parser.CsvReader;
import com.github.mkorman9.neural.network.NeuralNetwork;
import com.github.mkorman9.neural.network.reader.DefaultReader;
import com.github.mkorman9.neural.network.writer.DefaultWriter;
import com.github.mkorman9.neural.network.writer.Writer;
import com.google.common.collect.Lists;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {
    private static final int MAX_THREADS = 4;

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
        ExecutorService executor = Executors.newFixedThreadPool(MAX_THREADS);

        for (int i = 0; i < output.size(); i++) {
            Vector labels = output.row(0);
            executor.execute(new TrainingTask(input, i, labels));
        }

        executor.shutdown();
    }

    private static void predict(File inputFile, File outputFile) {
        Matrix input = new CsvReader().readFromFile(inputFile);
        Vector labels = new CsvReader().readFromFile(outputFile).column(0);
        List<NeuralNetwork> networks = Lists.newArrayList();

        for (int i = 0; i < 10; i++) {
            Model model = new DefaultReader().read(new File(String.format("target/model_%d.txt", i)));
            NeuralNetwork neuralNetwork = new NeuralNetwork(model, new SigmoidFunction());
            networks.add(neuralNetwork);
        }

        int successCount = 0;
        for (int i = 0; i < input.size(); i++) {
            Vector inputRow = input.row(i);
            double expectedAnswer = labels.get(i);
            List<Double> networksAnswers = Lists.newArrayList();

            for (int j = 0; j < 10; j++) {
                double prediction = networks.get(j).predict(inputRow);
                networksAnswers.add(prediction);
            }

            double max = 0.0;
            int maxIndex = -1;
            for (int j = 0; j < networksAnswers.size(); j++) {
                if (networksAnswers.get(j) >= max) {
                    max = networksAnswers.get(j);
                    maxIndex = j;
                }
            }

            if (expectedAnswer == maxIndex) {
                successCount += 1;
            }
        }

        System.out.printf("Guessed %d/%d (%f%%)\n", successCount, input.size(), (double) successCount / (double) input.size() * 100.0);
    }

    private static class TrainingTask implements Runnable {
        private final Matrix input;
        private final int n;
        private final Vector labels;

        public TrainingTask(Matrix input, int n, Vector labels) {
            this.input = input;
            this.n = n;
            this.labels = labels;
        }

        @Override
        public void run() {
            NeuralNetwork neuralNetwork = new NeuralNetwork(input.row(0).size(), new SigmoidFunction(), 1000);
            neuralNetwork.learn(input, labels);

            Writer writer = new DefaultWriter();
            writer.write(neuralNetwork.getModel(), new File(String.format("target/model_%d.txt", n)));

            System.out.printf("Wrote model for '%d' to target/model_%d.txt\n", n, n);
        }
    }
}
