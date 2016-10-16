package com.github.mkorman9.neural.data.parser;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Vector;
import com.google.common.collect.Lists;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CsvReader implements Reader {
    private static final String SEPARATOR = ",";

    @Override
    public Matrix readFromFile(File file) {
        List<Vector> result = Lists.newArrayList();
        try (Stream<String> fileStream = Files.lines(Paths.get(file.getAbsolutePath()))) {
            result = fileStream
                    .map(s -> Vector.create(Lists.newArrayList(s.split(SEPARATOR)).stream()
                                                                    .map(Double::valueOf)
                                                                    .collect(Collectors.toList())))
                    .collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return Matrix.create(result);
    }
}
