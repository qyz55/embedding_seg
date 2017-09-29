# Video Segmentation by pixel embedding

## Introduction

<!-- TODO -->

### Installation

1. Clone this repository

    ```Shell
    git clone git@github.com:meijieru/embedding_seg.git
    ```

1. Build

    - Ensure your CUDA configuration and c++11 support of your cpp compiler.
    - Modify `src/cpp/CMakeLists.txt`, change `TF_INC` according to your config. You could get this path by

        ```Shell
        $ python
        >>> import tensorflow as tf
        >>> tf.sysconfig.get_include()
        ```

    - Go to `src` and execute `make build`.

1. Train

Default config file is `experiments/config/default.json`. Modify config file and `makefile`, then `make train` in dir `src`.
