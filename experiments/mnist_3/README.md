# Spherical MNIST experiment

Data processing is adopted from ```https://github.com/jonkhler/s2cnn```.

## Generate the spherical MNIST data set

- __NR__: non rotated
- __R__: randomly rotated

##### train: __NR__ - test: __NR__
```bash
python gendata.py --no_rotate_train --no_rotate_test --output_file=NR-NR.gz
```

##### train: __R__ - test: __R__
```bash
python gendata.py --output_file=R-R.gz
```

##### train: __NR__ - test: __R__
```bash
python gendata.py --no_rotate_train --output_file=NR-R.gz
```

This will generate a `.gz` file in the same folder containing the compressed generated dataset.

To get more information about other params for the data generation (noise magnitude, number of images having the same random rotations etc.):
```bash
python3 gendata.py --help
```

## Note: Our training program will automatically do the S2-FFT to transform preprocessed data from the ```.gz``` file into out SO3vec format.
