import sklearn
import sklearn.datasets

if __name__ == '__main__':
    dataset = sklearn.datasets.make_moons(10000, noise=0.2)
    x_training = dataset[0].tolist()[:8000]
    y_training = dataset[1].tolist()[:8000]
    x_test = dataset[0].tolist()[8000:]
    y_test = dataset[1].tolist()[8000:]

    with open('../simple_training_input.txt', 'w') as f:
        f.write('\n'.join([' '.join([str(b) for b in a]) for a in x_training]))
    with open('../simple_training_output.txt', 'w') as f:
        f.write('\n'.join([str(a) for a in y_training]))
    with open('../simple_test_input.txt', 'w') as f:
            f.write('\n'.join([' '.join([str(b) for b in a]) for a in x_test]))
    with open('../simple_test_output.txt', 'w') as f:
        f.write('\n'.join([str(a) for a in y_test]))
