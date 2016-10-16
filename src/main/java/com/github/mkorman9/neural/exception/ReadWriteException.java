package com.github.mkorman9.neural.exception;

public class ReadWriteException extends RuntimeException {
    public ReadWriteException(String message, Throwable throwable) {
        super(message, throwable);
    }
}
