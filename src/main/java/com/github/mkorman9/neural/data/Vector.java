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
