*******************************************************************************************************************************************

                                           # Advanced-Computer-Architecture (ECEN 5593)

*******************************************************************************************************************************************

This course is taken by Dr. Dan Connors at University of Colorado, Boulder. In this course we learnt advanced concepts of Computer Architecture such as cache structure, branch predictors, multi-threaded GPU, GPU's memory hierarchy etc. It also teaches NVIDIA's CUDA programming and OpenCL for other GPUs. The last thing taught in this class was the ways to optimize the GPU.
This repository consists of all the assignments that I did in this course. 

• Assignment 1 is based on branch predictor using PIN Tool. It describes about 3 types of branch predictors basically 2-bit Counter, Global History Register, Global History Table, Per Address History Register, Global History Table. The analysis was done on 4 different benchmark Bzip2, PearlBench, Sjeng and gcc using the PIN Tool.

• Assignment 2 is an introduction to CUDA programming. It deals with vector reduce and add i.e. reducing the size of a 1D vector and adding them to get their addition in a faster way using shared memory and compare its result with it's implementation on CPU.

• Assignment 3 deals with implementation of 3 image processing filters viz. Sobel filter, Average filter and High boost filter on a GPU and analyse its execution time and compare it with CPU. We can see how fast a GPU could be for large image processing applications.

• Assignment 4 deals with the Histogram generation of random number sequence. It basically creates a list of N numbers between 0 to 9 and count how many times each number occurs in the list.

The professor also gave some reading assignments in which we were suppose to read some research papers on some topics and give a report of it stating what was the research about? What did the authors do? What did the achieve? What were the shortcomings and my input? These papers were extremely important and useful as they gave me a deeper understanding about the course.

•	Reading Assignment 1: Dynamic Branch predictors using perceptrons

•	Reading Assignment 2: An Algorithm for exploring multiple arithmetic units

•	Reading Assignment 3: A Checkpoint/Restart Scheme for CUDA Programs with Complex Computation States

•	Reading Assignment 4:  A Dynamic Compilation Framework for Controlling Microprocessor Energy and Performance

•	Reading Assignment 5:  An Analysis of Resilience Techniques for Exa-scale Computing Platforms

