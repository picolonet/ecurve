# cuda-fixnum for snark challenge

For each of mnt4 and mnt6, this takes the pairwise product of two arrays

See `main.cu` for the implementation

To build and run:

1. `./build.sh`
2. `./main compute inputs outputs`
3. `shasum outputs` should be `b0f4a59a4be1c878dd9698fae7f1be86d8261025`

you will need to edit /Makefile:GENCODES to match your GPU [see here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
