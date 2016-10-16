package com.github.mkorman9.neural.network.reader;

import com.github.mkorman9.neural.data.Model;

import java.io.File;

public interface Reader {
    Model read(File file);
}
