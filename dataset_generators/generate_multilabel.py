import sklearn
import sklearn.datasets

if __name__ == '__main__':
    dataset = sklearn.datasets.make_classification(n_samples=20000, n_features=5, n_informative=5,
                                                   n_redundant=0, n_repeated=0, n_classes=3)
    x_training = dataset[0].tolist()[:18000]
    y_training = dataset[1].tolist()[:18000]
    x_test = dataset[0].tolist()[18000:]
    y_test = dataset[1].tolist()[18000:]

    with open('../multilabel_training_input.txt', 'w') as f:
        f.write('\n'.join([' '.join([str(b) for b in a]) for a in x_training]))
    with open('../multilabel_training_output.txt', 'w') as f:
        f.write('\n'.join([str(a) for a in y_training]))
    with open('../multilabel_test_input.txt', 'w') as f:
            f.write('\n'.join([' '.join([str(b) for b in a]) for a in x_test]))
    with open('../multilabel_test_output.txt', 'w') as f:
        f.write('\n'.join([str(a) for a in y_test]))
