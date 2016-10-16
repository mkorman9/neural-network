package com.github.mkorman9.neural.network.writer;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;
import com.github.mkorman9.neural.exception.ReadWriteException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.stream.Collectors;

public class DefaultWriter implements Writer {
    private static final String LINE_SEPARATOR = "\n";
    private static final String DELIMITER = " ";

    @Override
    public void write(Model model, File file) {
        try (OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(file))) {
            writeDimension(model, writer);
            writeHiddenLayerWeights(model, writer);
            writeHiddenLayerBias(model, writer);
            writeOutputLayerWeights(model, writer);
            writeOutputLayerBias(model, writer);
            writer.close();
        } catch (IOException e) {
            throw new ReadWriteException("Error while writing file " + file.getAbsolutePath(), e);
        }
    }

    private void writeDimension(Model model, OutputStreamWriter writer) throws IOException {
        writeLine(writer, model.getHiddenLayerModel().getBias().size());
    }

    private void writeOutputLayerBias(Model model, OutputStreamWriter writer) throws IOException {
        writeLine(writer, model.getOutputLayerModel().getBias());
    }

    private void writeOutputLayerWeights(Model model, OutputStreamWriter writer) throws IOException {
        writeLine(writer, model.getOutputLayerModel().getWeights());
    }

    private void writeHiddenLayerBias(Model model, OutputStreamWriter writer) throws IOException {
        writeLine(writer, model.getHiddenLayerModel().getBias());
    }

    private void writeHiddenLayerWeights(Model model, OutputStreamWriter writer) throws IOException {
        writeMatrix(writer, model.getHiddenLayerModel().getWeights());
    }

    private void writeLine(OutputStreamWriter writer, String line) throws IOException {
        writer.write(line + LINE_SEPARATOR);
    }

    private void writeLine(OutputStreamWriter writer, int value) throws IOException {
        writeLine(writer, String.valueOf(value));
    }

    private void writeLine(OutputStreamWriter writer, double value) throws IOException {
        writeLine(writer, String.valueOf(value));
    }

    private void writeLine(OutputStreamWriter writer, Vector value) throws IOException {
        writeLine(writer, value.values().stream()
                                        .map(String::valueOf)
                                        .collect(Collectors.joining(DELIMITER))
        );
    }

    private void writeMatrix(OutputStreamWriter writer, Matrix value) throws IOException {
        for (int i = 0; i < value.size(); i++) {
            writeLine(writer, value.row(i));
        }
    }

}
