# Pricing of out-of-the-money Barrier Options Using Importance Sampling

This is a [UROPS](https://www.science.nus.edu.sg/undergraduates/undergraduate-research/urops/) (Undergraduate Research Opportunities Programme in Science) Project done under the supervision of [Prof. Julian Sester](https://sites.google.com/view/juliansester/home) at the National University of Singapore.

 Barrier options are path-dependent exotic derivatives similar to vanilla options, 
 but are only activated or deactivated once the price of the underlying asset
 reaches a certain level, which is called the barrier. For this project, we first
 derive the formula for the theoretical price of a barrier option exercised in the
 European style. Next, we shall try to price the barrier option using Monte Carlo
 simulations and compare the simulated price to the theoretical price. However,
 when it comes to out-of-the-money barrier options, the likelihood of hitting the
 barrier is low, and thus, most simulations do not contribute meaningful information. 
 This is computationally inefficient. Therefore, importance sampling
 is applied to alter the probability distribution of the underlying asset’s price
 path to increase the probability of the option’s barrier being hit, reducing the variance and improving the simulation's accuracy.

My thesis can be found at [Thesis-Importance_Sampling.pdf](https://github.com/lee-wei-xuan/barrier_option_pricing/blob/main/Thesis-Importance_Sampling.pdf).

If you are interested in the code for Monte Carlo simulations ***without*** importance sampling, definitely check out [Direct_Method.py](https://github.com/lee-wei-xuan/barrier_option_pricing/blob/main/Direct_Method.py), whereas the one ***with*** importance sampling can be found at [Importance_Sampling.py](https://github.com/lee-wei-xuan/barrier_option_pricing/blob/main/Importance_Sampling.py). 

Lastly, you can also find the slides for my final presentation at [UROPS_Presentation.pdf](https://github.com/lee-wei-xuan/barrier_option_pricing/blob/main/UROPS_Presentation.pdf).



