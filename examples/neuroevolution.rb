# Make sure the gem is installed first with `gem install machine_learning_workbench`
# Alternatively, add `gem 'machine_learning_workbench'` to your Gemfile if using Bundle,
# followed by a `bundle install`
require 'machine_learning_workbench'
# Workbench shorthands
XNES = WB::Optimizer::NaturalEvolutionStrategies::XNES
FFNN = WB::NeuralNetwork::FeedForward

# Let's address the XOR problem, as it requires nonlinear fitting
XOR = {[0,0] => 0, [1,0] => 1, [0,1] => 1, [1,1] => 0}
# A classic [2,2,1] feed-forward network will do: 2 inputs, 2 hidden, 1 output
# For other uses, make sure you match the first number to the number of inputs, and
# the last one as the number of outputs; then add as many layers as needed, by
# specifying the size of each. Here we have only one, of size 2.
# NOTE: If this totals thousands of weights, you may want to switch to SNES or BDNES
# for speed. In the second case, use the function `nweights_per_layer` when instantiating
# BDNES rather than `nweights`.
NET = FFNN.new [2,2,1], act_fn: :logistic
# Note: the process is exactly the same, from instantiation to training, for recurrent
# networks using the class `WB::NeuralNetwork::Recursive`.
# Of course RNNs should be applied to sequential tasks, while XOR is static

# We will search for the network's weights with a black-box optimization algorithm
# This means we will search for arrays of numbers, which need to be scored.
# The scoring process will work as follows: use the numbers as weights for the neural
# network, test the network on classifying the 4 cases of XOR, use that count as the
# score for the weights (original array of numbers).

# Hence the fitness looks as follows:
def fitness weights
  # Each list of weights uniquely defines a neural network
  NET.load_weights weights
  # Activate the network on each of the XOR instances
  # - prediction: the output of the network
  # - observation: correct value, our target
  pred_obs = XOR.map do |input, obs|
    # The network can have an arbitrary number of output neurons
    # Since here we have only one, we extract the value calling `#first`
    output = NET.activate(input).first
    # Here we interpret the output as classification
    pred = output > 0.5 ? 1 : 0
    # Finally accumulate prediction-observation pairs
    [pred, obs]
  end
  # To build a score out of this, we count the number of correct classifications
  score = Float(pred_obs.count { |pr, ob| pr == ob })
  # That's it, this will score the weights based on their network's performance
end

# Next comes initializing the black-box stochastic optimization algorithm
# We are searching for the network's weights, this gives us the search space dimensionality
# We'll use XNES as we are working with less than 100 dimensions (weights)
nes = XNES.new NET.nweights, method(:fitness), :max, rseed: 15
# Note: the random seed is fixed here to ensure the task is solved in one try in few iterations
# In a real task, best using an over-large network, more iterations, and try several seeds

# NOTE: In practical applications it is best to delegate parallelization to the fitness
# function instead of computing the fitness of one individual at a time. This can be
# achieved by passing  an objective function defined on a _list_ of weight-lists, and
# setting the `parallel_fit` switch to `true`:
# nes = XNES.new NET.nweights,
#   -> (genotypes) { Parallel.map genotypes, &method(:fitness) },
#   :max, rseed: 15, parallel_fit: true


# Nothing left but to run the optimization algorithm, few epochs here will suffice
50.times { nes.train }
# OK! now remember, `NET` currently holds the weights of the last evaluation
# Let's fetch the best individual found so far
best_fit, best_weights = nes.best
# Let's run them again to check they work
result = fitness best_weights # careful here if you defined a parallel `fitness`
puts "The found network achieves a score of #{result} out of 4 in the XOR task"
puts "Weights: #{best_weights}"
puts "Done!"
