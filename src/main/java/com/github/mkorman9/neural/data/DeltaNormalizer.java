package com.github.mkorman9.neural.data;

public class DeltaNormalizer {
    public static double normalize(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0;
        }

        return value;
    }
}
