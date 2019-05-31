# cuda-fixnum for snark challenge

For each of mnt4 and mnt6, this takes the pairwise product of two arrays. That is, it maps over two arrays.

See `main.cu` for the implementation

To build and run:

1. `./build.sh`
2. `./main compute inputs outputs`
3. `shasum outputs` should be `b0f4a59a4be1c878dd9698fae7f1be86d8261025`

you will need to edit /Makefile:GENCODES to match your GPU [see here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

To complete the [first part of stage 1](https://coinlist.co/build/coda/pages/problem-01-field-arithmetic), change this map to a reduce, and match the [reference](https://github.com/CodaProtocol/snark-challenge/tree/master/reference-01-field-arithmetic). See [this example of a reduce using cuda](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction) to get started.
