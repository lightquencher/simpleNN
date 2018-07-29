# simpleNN
This is my first attempt at making a neural network

In my code I decided to create a class that is responsible for an individual neuron and can apply a sum and a sigmoid. Then a class for running the network through once so that I can keep all the values inbetween accesible for weight adjusting. Then there is a loop that is responsible for complete loops of the complete database and loops inside for training and testing. In training it calls adjust() which then adjusts the weights as explained below. There is also a plot feature at the end of the code to visualise the dataset.

### Adjusting weights
The system I have implemented now is that it runs the network with one set of inputs, then using (what I think is) backpropagation to see the change in error (cost) by the change in the very first weight to adjust the weight slightly. Then repeats the network with same inputs but different weights and adjusts the next weight. This continues until all the weights are changed for that specific input and repeats for next input.
I am going to try change only the least significant weight at a time, later.

## Progress
Currently I can train it and it adjusts weights, but after a certain number of loops it messes up the weights and it starts to always guess the same exact answer. I don't know if this is because i'm adjusting weights incorrectly or if the learning rate is too high or low. I am quite new to this and so I do not fully understand what is going wrong.




## Errors that occur
### Error 1
Code's Guess : Answer : Error(cost)

1.000 : 1 : 0.000

0.000 : 0 : 0.000

0.945 : 0 : 0.893

0.000 : 1 : 1.000

Then after this it just continues to guess 0, and only 0

### Error 2
Code's Guess : Answer : Error(cost)

1.00 : 1 : 0.000

0.00 : 0 : 0.000

0.00 : 0 : 0.000

0.25 : 1 : 0.563

1.00 : 1 : 0.000

The learning rate at this point was 0.5, which makes a little more sense where the 0.25 came from

## Requirements:
Numpy
Matplotlib
