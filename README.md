# Integral Equation DenseNet-summer2021
This repository contains preliminary implementations of a new architecture of neural network. The model can be described as a continuous version of DenseNet, and it utilizes an integral solver in order to produce output. For details about the architecture, please contact Dr. Stephen Scott, Dr. Mikil Foss, or Dr. Petronela Radu of the University of Nebraska-Lincoln. For questions about the implementation of this repo, contact Eylon Caplan at eyloncaplan1@gmail.com.

## File Summaries
All models in this repo are based on the earlier works of Dr. Foss, which can be found in "FossModels". The early versions found in "EarlyVersions" are progressive attempts at converting Foss's models into a model that uses tensorflow and a computational graph. All currently useful files can be found in "LaterVersions".

Their contents are summarized here:

NoiseCorrectionTest.ipynb: This is the best notebook to start on when trying to understand the code. Because the model itself is mostly repeated between files, NoiseCorrectionTest is the only one in which the model is commented. After defining the model, the notebook proceeds to test its improvement on a noise reduction dataset. An array of 0's and 1's is produced as the target (y). It is then perturbed with some noise to create an input (x). The notebook trains on 1000 data with depth of 5, and is later trained on 10,000 data on the same depth. The results can be seen in the graphs.

LoopedGradient_7-30-21-ModelVersion.ipynb: This is the first working version of the model that can be iterated and tested. The model is trained to output the input array while being fed the same input each time. It is then trained to output some set random target array while being fed the same input each time. Finally, it is trained to output some set random target array while being fed different inputs each iteration. The results can be seen in the graphs.

AnalyticTest.ipynb: This was a failed attempt at testing the model analytically. The idea was to pick a function such as U(t) = t^2, pick parameters for the parameter filter, and to compute what the bias would have to be in order to satisfy the equation. Then, train and see if the bias is tuned to these values. This did not work. However, there are several likely reasons that it did not work. Firstly, the issue of discretization/depth has not been addressed yet (it will be mentioned later). Secondly, the test was likely not implemented correctly. The network should not be allowed to train. Rather, the bias and parameter filter should be set to values such that they solve the equation with U(t) = the function (such as t^2). The test should then check to see that the output of each layer approximates U(that layer). 

NoiseCorrectionInputSizes.ipynb: This is the same as NoiseCorrectionTest but attempts some trials on larger data (the size of MNIST)

IntegralDenseNetClass.ipynb: This is a very incomplete attempt to convert the working model found in NoiseCorrectionTest into a class. It is absolutely not functioning yet. The idea is to have a .fit function that creates and optimizes the parameters according to the data. Separately, one can have a feedForward function that only takes in input to the network and uses the known parameters to generate output. Thus the .fit function can prepare the inputs and targets, and then call the optimize function, whose job it is to create a gradient tape, run the feedForward, track the gradients, and apply them to the parameters. Currently this class contains only a very rough framework for a working class. Also, the line that converts a function into a tensorflow function must be called at some point so that it can be GPU optimized.

## To Do
There is much work to be done in this project.
1. Finish the class so that use of the model is compact and easy to use
2. Run tests on the GPU to make sure that it is faster than on a CPU
3. Separate the discretization value dt from the parameter depth t1/t0. Currently dt is coded as the inverse of t1, but should be a hyperparameter in its own right that does not depend on the parameter depth.
4. Attempt batch training. The dimensions of the tensors should already be set up for batches. However, the dataset is currently being fed in one datum at a time. The change should be easy but it is likely that there is some assumption of batch=1 hard coded in somewhere that must be corrected.
3. Add a dense layer to the end of the feedForward to allow for classification instead of only regression
4. Perform the analytic test as described above. However, note that the feed forward must be performed eagerly in order to be able to look at the layers' values at intermediary steps. To do this, use the actual function name (uSoln) instead of the tensorflow function name (TFuSoln).
5. Once classification works, test on datasets like MNIST and CIFAR
6. Try other integration techniques apart from the trapezoidal rule (e.g. Simpson's rule)
7. Conduct some grid searches with hyperparameters

## Notes
Sometimes when running the code in a jupyter notebook, one will get a tensorflow warning as seen in In[45] of LoopedGradient_7-30-21-ModelVersion.ipynb. I am relatively sure that this is because each new call to the function causes retracing. However, this is because the notebook stores the previous information about the function. If run all at once, this warning should never be reproduced.


