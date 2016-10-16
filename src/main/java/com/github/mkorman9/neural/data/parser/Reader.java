package com.github.mkorman9.neural.data.parser;

import com.github.mkorman9.neural.data.Matrix;

import java.io.File;

public interface Reader {
    Matrix readFromFile(File file);
}
