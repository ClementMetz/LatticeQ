# LatticeQ

This is the user manual for LatticeQ source code from the paper Lattice Quantization https://ieeexplore.ieee.org/document/10137188
(NeurIPS 2022 ML for Systems workshop, DATE 2023).

## Requirements

The prerequisites are the following packages :
* pytorch (>1.8)
* torchvision (>0.9)
* numpy
* tqdm

You might want to install them in a dedicated anaconda environment.

## Code overview
The code should include 7 files :
* main_quant.py contains the main function
* quantize_filters.py which contains functions useful for quantization
* util which contains auxiliary functions for score computation, display and block reconstruction utilities
* torchTransformer which contains the function to setup quantized model overhead
* QConv2d, QLinear and QPooling contain the overhead of torch.nn classes to quantize weights and/or activations

## Experiments
Here is how to run a short version of our Resnet18 per-channel quantization to W4A8 with
bias correction :

`
python main_quant.py --batch_size=512 --model_name=resnet18 --bitsW=4 --quant_a --bitsA=8 --quant_basis --bitsB=8 --quant_a_type=per_layer --nb_restarts=5 --steps=80 --bias_corr --valdir=myvalidationdatapath
`

* batch_size is the size of the calibration set used for activations thresholds.
* model_name is the name of the model as written in torchvision.models library.
* bitsW is the bitwidth used for weights in all but first and last layers.
* quant_a needs to be stated when you want to quantize activations.
* bitsA is the bitwidth used for activations in all but first, last, and pooling layers
* quant_a_type is the type of activation quantization, either per_layer (naive) or per_channel (quantile based).
* nb_restarts is the number of restarts used for random search with restarts.
* steps is the number of steps for the random search. We used between 800 and 1500 steps
to perform our experiments, but this might result in several hours long runs,
especially for very deep models such as densenet.
* bias_corr needs to be stated when you want to use bias correction for weights.
* valdir is the path of your validation set.

Here is how to run our Resnet18 per-layer quantization to W4A32 with blockwise reconstruction :

python main_quant.py --batch_size=512 --model_name=resnet18 --bitsW=4 --per_layer --block_reconstruction --epochs=2000 --nb_restarts=5 --steps=800 --valdir=myvalidationdatapath

* block_reconstruction triggers the block reconstruction step (level 2 data consuming enhancement)
* epochs controls the number of training epochs for each block

Only Resnet18 and PreResnet18 are currently supported for block reconstruction.

## Results

Please keep in mind that the results obtained might not exactly be those indicated in the
paper, this is because of the randomness of our optimization strategy and calibration data. For the data free approach, since the search space of quantization bases is non trivial, we encourage any attempt at finding efficient combinations of steps and restarts. These results are produced in the same conditions as the data free enhancement results from Table 3 of the paper. Please use the following commandline to reproduce these results.

* `python main_quant.py --batch_size=512 --model_name=___ --bitsW=_ --quant_a --bitsA=_ --quant_basis --bitsB=_ --quant_a_type=___ --nb_restarts=5 --steps=800 --bias_corr --valdir=myvalidationdatapath`



Network | W4A8B8 | W4A4B4 | W3A8B8
---|---|---|---
Resnet-18 | 68.910 | 66.952 | 66.820

The following table is the result of a short experiment with only 80 steps of random search. This version only takes ca. 2 minutes to run.

* `python main_quant.py --batch_size=512 --model_name=___ --bitsW=_ --quant_a --bitsA=_ --quant_basis --bitsB=_ --quant_a_type=___ --nb_restarts=5 --steps=80 --bias_corr --valdir=myvalidationdatapath`



Network | W4A8B8 | W4A4B4 | W3A8B8
---|---|---|---
Resnet-18 | 69.042 | 66.520 | 65.632


## Observations
You might want to use CUDA_VISIBLE_DEVICES to precise the gpu you wish to use for the
experiment.

You might want to use calibration data outside of your validation set, in this case
you need to precise --traindir=mycalibrationdatapath.

You should state the model name as it is written in torchvision.models library, or preresnet18 to use the model from Brecq's repository.

Our source code is not optimized for memory consumption yet, and it does not implement
multi gpu. Hence, you might want to use other sizes of calibration batches, depending on
your machine and the network. The results should not be dramatically different. In principle, every
model we tested (except VGG) should run on a Nvidia RTX 2080 with a batch size of 256.


