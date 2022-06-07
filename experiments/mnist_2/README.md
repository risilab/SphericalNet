# Spherical MNIST experiment

Data processing is adopted from ```https://github.com/jonkhler/s2cnn```.

## Generate the spherical MNIST data set

- __NR__: non rotated
- __R__: randomly rotated

##### train: __NR__ - test: __NR__
```bash
python3 gendata.py --no_rotate_train --no_rotate_test
```

##### train: __R__ - test: __R__
```bash
python3 gendata.py
```

##### train: __NR__ - test: __R__
```bash
python3 gendata.py --no_rotate_train
```

This will generate a `s2_mnist.gz` in the same folder containing the compressed generated dataset.

To get more information about other params for the data generation (noise magnitude, number of images having the same random rotations etc.):
```bash
python3 gendata.py --help
```

## Convert the preprocessed data into our SO3vec format via S2-FFT

```python convert_data_to_SO3vec.py```

Change the variable ```maxl``` as the bandwidth for S2-FFT. The default value is 11. The S2 Fourier transformed input data for our models is saved into ```spherical_mnist_maxl_11.pkl``` (by default).

