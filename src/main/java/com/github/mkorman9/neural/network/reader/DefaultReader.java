package com.github.mkorman9.neural.network.reader;

import com.github.mkorman9.neural.data.*;
import com.github.mkorman9.neural.exception.ReadWriteException;
import com.github.mkorman9.neural.network.NeuralNetwork;
import com.google.common.collect.Lists;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;

public class DefaultReader implements Reader {
    private static final String DELIMITER = " ";

    @Override
    public Model read(File file) {
        Matrix hiddenWeights;
        Vector hiddenBias;
        Vector outputWeights;
        double outputBias;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
            int dimension = readInt(reader);
            hiddenWeights = readMatrix(reader, dimension);
            hiddenBias = readVector(reader, dimension);
            outputWeights = readVector(reader, dimension);
            outputBias = readDouble(reader);
            reader.close();
        }
        catch (IOException e) {
            throw new ReadWriteException("Error while reading file " + file.getAbsolutePath(), e);
        }

        return new Model(
                new HiddenLayerModel(hiddenWeights, hiddenBias),
                new OutputLayerModel(outputWeights, outputBias)
        );
    }

    private double readDouble(BufferedReader reader) throws IOException {
        return Double.valueOf(reader.readLine());
    }

    private Vector readVector(BufferedReader reader, int dimension) throws IOException {
        return Vector.create(Lists.newArrayList(reader.readLine().split(DELIMITER)).stream()
                .map(Double::valueOf)
                .collect(Collectors.toList()));
    }

    private Matrix readMatrix(BufferedReader reader, int dimension) throws IOException {
        List<Vector> rows = Lists.newArrayList();
        for (int i = 0; i < dimension; i++) {
            rows.add(readVector(reader, dimension));
        }
        return Matrix.create(rows);
    }

    private int readInt(BufferedReader reader) throws IOException {
        return Integer.valueOf(reader.readLine());
    }
}
