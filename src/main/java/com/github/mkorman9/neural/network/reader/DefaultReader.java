package com.github.mkorman9.neural.network.reader;

import com.github.mkorman9.neural.data.*;
import com.github.mkorman9.neural.exception.ReadWriteException;
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
        Matrix outputWeights;
        Vector outputBias;
        int inputsCount = 0;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
            inputsCount = readInt(reader);
            int hiddenNeurons = readInt(reader);
            hiddenWeights = readMatrix(reader, inputsCount);
            hiddenBias = readVector(reader);
            outputWeights = readMatrix(reader, hiddenNeurons);
            outputBias = readVector(reader);
            reader.close();
        }
        catch (IOException e) {
            throw new ReadWriteException("Error while reading file " + file.getAbsolutePath(), e);
        }

        return new Model(
                inputsCount,
                new HiddenLayerModel(hiddenWeights, hiddenBias),
                new OutputLayerModel(outputWeights, outputBias)
        );
    }

    private Vector readVector(BufferedReader reader) throws IOException {
        return Vector.create(Lists.newArrayList(reader.readLine().split(DELIMITER)).stream()
                .map(Double::valueOf)
                .collect(Collectors.toList()));
    }

    private Matrix readMatrix(BufferedReader reader, int dimension) throws IOException {
        List<Vector> rows = Lists.newArrayList();
        for (int i = 0; i < dimension; i++) {
            rows.add(readVector(reader));
        }
        return Matrix.create(rows);
    }

    private int readInt(BufferedReader reader) throws IOException {
        return Integer.valueOf(reader.readLine());
    }
}
