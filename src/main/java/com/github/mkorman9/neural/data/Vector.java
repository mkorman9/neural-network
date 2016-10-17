package com.github.mkorman9.neural.data;

import com.google.common.collect.Lists;

import java.util.List;

public class Vector {
    private List<Double> values;

    private Vector(List<Double> values) {
        this.values = values;
    }

    public int size() {
        return values.size();
    }

    public Double get(int i) {
        return values.get(i);
    }

    public void set(int i, Double value) {
        values.set(i, value);
    }

    public List<Double> values() {
        return values;
    }

    public Matrix multiply(Vector other) {
        List<Vector> rows = Lists.newArrayList();
        for (int i = 0; i < values.size(); i++) {
            List<Double> columns = Lists.newArrayList();
            for (int j = 0; j < other.values.size(); j++) {
                columns.add(values.get(i) * other.values.get(j));
            }
            rows.add(Vector.create(columns));
        }
        return Matrix.create(rows);
    }

    public static Vector create(Double... values) {
        return new Vector(Lists.newArrayList(values));
    }

    public static Vector create(List<Double> values) {
        return new Vector(Lists.newArrayList(values));
    }

    public static Vector zero(int size) {
        List<Double> values = Lists.newArrayList();
        for (int i = 0; i < size; i++) {
            values.add(0.0);
        }
        return new Vector(values);
    }

    public static Vector random(int size) {
        List<Double> values = Lists.newArrayList();
        for (int i = 0; i < size; i++) {
            values.add(RandomValue.generate());
        }
        return new Vector(values);
    }
}
