package com.github.mkorman9.neural;

import com.github.mkorman9.neural.activation.SigmoidFunction;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;
import com.github.mkorman9.neural.network.NeuralNetwork;
import com.github.mkorman9.neural.parser.CsvReader;
import com.github.mkorman9.neural.parser.Reader;

import java.io.File;

public class Main {
    public static void main(String[] args) {
        // read training data from files
        Reader filesReader = new CsvReader();
        Matrix input = filesReader.readFromFile(new File(args[0]));
        Vector output = filesReader.readFromFile(new File(args[1])).column(0);

        // train network
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, new SigmoidFunction(), 1000);
        neuralNetwork.learn(input, output);

        // try to predict result for new input data
        System.out.println(neuralNetwork.predict(Vector.create(-1.2, 0.796)));
    }
}
