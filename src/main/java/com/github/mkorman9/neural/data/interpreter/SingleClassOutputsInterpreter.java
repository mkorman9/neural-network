package com.github.mkorman9.neural.data.interpreter;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;

public class SingleClassOutputsInterpreter {
    public Vector interpret(Matrix outputs) {
        return outputs.column(0);
    }
}
