package com.github.mkorman9.neural.network.writer;

import com.github.mkorman9.neural.data.Model;

import java.io.File;

public interface Writer {
    void write(Model model, File file);
}
