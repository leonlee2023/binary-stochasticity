This code package is used to simualted the binary stochastic deep neural network.
-- Features of this code pakage:
    - Stochastic binarization of the forwarding signals in each layer of a neural network;
    - Stochastic binarization of the activation derivatives in each layer;
    - Sign of backpropagating errors in each layer;
    - Weight quantization during the training;
    - Supports simulation of crossbar array of memristors as the synaptic array for in-situ deep learning:
        Memeristor has tunable conductance, which mimiks the plasticity of biological synapse.
        Crossbar array of memristors can perform the Vector-Matrix Multiplication in one-step using Ohm's law and Kirchhoff's current law. 
        However, applying accurate voltages and accurately sensing the current is trouble some. 
        In addition, memristors are noisy. Under identical potentiation pulses and depression pulses, the conductance changes is high nonlinear and fluctuated, thus preventing the in-situ deep learning. 
        In this work, these problems are sloved.
    - Better-than-base-line performance is obtained in memristor based deep learning system. 

More detials of this work should be addressed to our Paper titled "Stochastically binarized forwarding and error sign backpropagation empowered highly efficient deep learning" by Yang Li et al.

Developed by Dr. Wei Wang and Dr. Yang Li @ Shenzhen, China. Feb. 2023.

Language: Matlab
Software version: MATLAB R2021b, R2022b.

The code can be run on CPUs but it would be faster if you have Nvidia GPUs that support CUDA.
-- Type "gpuDevice" in Matlab command window to check if you have GPUs available in MATLAB
    - If an Error pops, then you don't aviable GPUs in MATLAB. 
        You can still run this code but with slower speed.
        If you believe you have Nvidia GPUs that support CUDA you computer, but have this error. You need to install the Nvidia drivers for CUDA (https://developer.nvidia.com/cuda-toolkit). 
    - If it shows "CUDADevice with properties: ....", then you can accelearte this code by the GPUs.


To start, you first need to download the MNIST dataset and CIFAR10 dataset, and preprocessing them.
-- MNIST (Website: http://yann.lecun.com/exdb/mnist/)
    - You need to download four files: 
        train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
        train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
        t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
        t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
    
    - Unzip them and copy unziped files to folder: "/dataset/mnist/".
    
    - Browse the working folder of Matlab to the folder "/dataset/mnist/"
        Run the code file "loadMNIST.m", then a "MNIST.mat" file is generated, and some example handwritten digits will be displayed. 

-- CIFAR10 (Website: http://www.cs.toronto.edu/~kriz/cifar.html)
    - Download the Matlab version of the dataset: 
        CIFAR-10 Matlab version	175 MB	md5sum: 70270af85842c9e89bb428ec9976c926

    - Unzip the downloaded file and copy the unziped ".mat" files to the flolder "/dataset/cifar10/"
    
    - Browse the working folder of Matlab to the folder "/dataset/cifar10/"
        Run the code file "loadCIFAR10.m", then a "cifar10.mat" file is generated, and some example images will be displayed. 

-- Fully connected neural networks for MNIST dataset
    - Browse the working folder of Matlab to the folder "/dnn_stochastic_mnist/"

    - Run the code file "dnn_fcn_train.m": the neural network training will be started.
        Key information during the training will be displayed in the command window.
        After the training finished, the cross-entropy and the recogniton accuracy during the training will be plotted.
        Three inference methods will be tested after the training finished: high precision, deterministic binarization, stochastic binarization.
        A ".mat" file will be generated which saves all the variables for the neural network setting, the croo-entropy and test accuracy during training, learned weights. 
    
    - Change the parameters in the code file "dnn_fcn_train.m", and re-run this file.
        To verify the effect of neural network parameters on its performance.
    
    - Subfolder "/dnn_stochastic_mnist/activation_logistic_parameter/"
        The codes to the loop of different prefactors "a" in the logistic activation function "z=1/(1+exp(-a*y))" under different learning schemes.
    
    - Subfolder "/dnn_stochastic_mnist/activation_types/"
        The codes to the loop of activation functions and derivatives under different learning schemes.

    - Subfolder "/dnn_stochastic_mnist/weight_integer/"
        The codes for training the neural network with weights of various types of integers.

    - Subfolder "/dnn_stochastic_mnist/weight_integer/"
        The codes for training the neural network using real memristor synaptic behaviors under idential potentiation and depression pulses. 

-- Convolutional neural networks for MNIST dataset
    - Browse the working folder of Matlab to the folder "/dcnn_stochastic_mnist/"

    - Run the code file "dnn_conv_train.m": the neural network training will be started.
        Key information during the training will be displayed in the command window.
        After the training finished, the cross-entropy and the recogniton accuracy during the training will be plotted.
        A ".mat" file will be generated which saves all the variables for the neural network setting, the croo-entropy and test accuracy during training, learned weights. 
    
    - Change the parameters in the code file "dnn_conv_train.m", and re-run this file.
        To verify the effect of neural network parameters on its performance.

    - Other code files to post processing the learned neural network:
        "inference.m": test different inference methods.
        "plot_weight_maps.m": plot the weight distribution.
        "train_results_comparison.m": compare training results for different setting or parameters.

-- Convolutional neural networks for CIFAR10 dataset
    - Browse the working folder of Matlab to the folder "dcnn_stochastic_cifar10/"

    - Run the code file "dnn_conv_train_HP.m": the neural network training will be started.
        (Warning: The training will takes approximately 20 hours with a single Nvidia RTX A4000 GPU, and approximately 100 hours without GPUs.)
        Key information during the training will be displayed in the command window.
        After the training finished, the cross-entropy and the recogniton accuracy during the training will be plotted.
        A ".mat" file will be generated which saves all the variables for the neural network setting, the croo-entropy and test accuracy during training, learned weights. 
    
    - Run the code file "dnn_conv_train_BS.m": the neural network training will be started.
        (Warning: The training will takes approximately 20 hours with a single Nvidia RTX A4000 GPU, and approximately 100 hours without GPUs.)
        Key information during the training will be displayed in the command window.
        After the training finished, the cross-entropy and the recogniton accuracy during the training will be plotted.
        A ".mat" file will be generated which saves all the variables for the neural network setting, the croo-entropy and test accuracy during training, learned weights. 
    
    - Other code files to post processing the learned neural network:
        "inference.m": test different inference methods.
        "train_results_comparison.m": compare training results for different setting or parameters.
