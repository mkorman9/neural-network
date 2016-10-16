package com.github.mkorman9.neural.data;

import com.google.common.collect.Lists;

import java.util.List;
import java.util.stream.Collectors;

public class Matrix {
    private List<Vector> rows;

    private Matrix(List<Vector> rows) {
        this.rows = rows;
    }

    public int size() {
        return rows.size();
    }

    public Vector row(int i) {
        return Vector.create(rows.get(i).values());
    }

    public Vector column(int j) {
        return Vector.create(rows.stream()
                                .map(r -> r.get(j))
                                .collect(Collectors.toList()));
    }

    public Double value(int i, int j) {
        return rows.get(i).get(j);
    }

    public void setValue(int i, int j, Double value) {
        rows.get(i).set(j, value);
    }

    public static Matrix create(Vector... rows) {
        return new Matrix(Lists.newArrayList(rows));
    }

    public static Matrix create(List<Vector> rows) {
        return new Matrix(Lists.newArrayList(rows));
    }

    public static Matrix random(int rows, int columns) {
        List<Vector> values = Lists.newArrayList();
        for (int i = 0; i < rows; i++) {
            values.add(Vector.random(columns));
        }
        return new Matrix(values);
    }

    public static Matrix zero(int rows, int columns) {
        List<Vector> values = Lists.newArrayList();
        for (int i = 0; i < rows; i++) {
            values.add(Vector.zero(columns));
        }
        return new Matrix(values);
    }
}
