package com.github.mkorman9.neural.data;

public class Model {
    private HiddenLayerModel hiddenLayerModel;
    private OutputLayerModel outputLayerModel;
    private int inputsCount;

    public Model(int inputsCount, HiddenLayerModel hiddenLayerModel, OutputLayerModel outputLayerModel) {
        this.inputsCount = inputsCount;
        this.hiddenLayerModel = hiddenLayerModel;
        this.outputLayerModel = outputLayerModel;
    }

    public HiddenLayerModel getHiddenLayerModel() {
        return hiddenLayerModel;
    }

    public void setHiddenLayerModel(HiddenLayerModel hiddenLayerModel) {
        this.hiddenLayerModel = hiddenLayerModel;
    }

    public OutputLayerModel getOutputLayerModel() {
        return outputLayerModel;
    }

    public void setOutputLayerModel(OutputLayerModel outputLayerModel) {
        this.outputLayerModel = outputLayerModel;
    }

    public int getInputsCount() {
        return inputsCount;
    }

    public void setInputsCount(int inputsCount) {
        this.inputsCount = inputsCount;
    }

    public int getHiddenLayerNeuronsCount() {
        return hiddenLayerModel.getBias().size();
    }

    public int getOutputLayerNeuronsCount() {
        return outputLayerModel.getBias().size();
    }
}
