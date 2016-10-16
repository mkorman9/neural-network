package com.github.mkorman9.neural.network;

class PredictionSuccessChecker {
    private static final double BORDER_OF_TRUE = 0.5;

    public boolean check(double value) {
        return value >= BORDER_OF_TRUE;
    }
}
