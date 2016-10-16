package com.github.mkorman9.neural.data.interpreter;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;

import java.util.List;

public class MultiClassOutputsInterpreter {
    public Matrix interpret(Matrix outputs, List<Double> classes) {
        Vector labels = outputs.column(0);
        Matrix result = Matrix.random(classes.size(), labels.size());
        for (int i = 0; i < labels.size(); i++) {
            for (int j = 0; j < classes.size(); j++) {
                result.setValue(j, i, labels.get(i).equals(classes.get(j)) ? 1.0 : 0.0);
            }
        }
        return result;
    }
}
