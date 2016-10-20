package com.github.mkorman9.neural.network.transformer;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;
import com.google.common.collect.Lists;

import java.util.List;

/*
    Conversion from plain labels:
     1
     2
     3
     4
    to feature vectors:
     1 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 1
*/
public class MulticlassOutputTransformer {
    private List<Double> classes;

    public MulticlassOutputTransformer(Double... classes) {
        this.classes = Lists.newArrayList(classes);
    }

    public Matrix transform(Vector output) {
        List<Vector> rows = Lists.newArrayList();
        for (int i = 0; i < output.size(); i++) {
            Vector row = Vector.zero(classes.size());
            for (int j = 0; j < classes.size(); j++) {
                if (output.get(i).equals(classes.get(j))) {
                    row.set(j, 1.0);
                }
            }
            rows.add(row);
        }
        return Matrix.create(rows);
    }
}
