package com.github.mkorman9.neural.data;

public class Model {
    private HiddenLayerModel hiddenLayerModel;
    private OutputLayerModel outputLayerModel;

    public Model(HiddenLayerModel hiddenLayerModel, OutputLayerModel outputLayerModel) {
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
}
