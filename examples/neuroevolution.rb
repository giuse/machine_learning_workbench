# Make sure the gem is installed first with `gem install machine_learning_workbench`
# Alternatively, add `gem 'machine_learning_workbench'` to your Gemfile if using Bundle,
# followed by a `bundle install`
require 'machine_learning_workbench'
# Workbench shorthands
XNES = WB::Optimizer::NaturalEvolutionStrategies::XNES
FFNN = WB::NeuralNetwork::FeedForward

# Let's address the XOR problem, as it requires nonlinear fitting
XOR = {[0,0] => 0, [1,0] => 1, [0,1] => 1, [1,1] => 0}
# A classic [2,2,1] (2 inputs, 2 hidden neurons, 1 output neurons) feed-forward
# network with nonlinear activations can solve this problem.
# To approximate more complex functions, keep the number of inputs and outputs
# fixed (they depend on the problem) and increase the number and/or size of
# hidden neurons. For example: [2, 10, 7, 4, 1].
# NOTE: If your network grows above few thousands of weights, XNES may be too slow.
# Try using SNES for large shallow networks or BDNES for deep networks.
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
    # Since here we have only one, we extract the value as the output
    output = NET.activate(input)[0]
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
nes = XNES.new NET.nweights, method(:fitness), :max, rseed: 0
# Note BDNES requires `NET.nweights_per_layer` rather than `NET.nweights` in initialization:
# nes = WB::Optimizer::NaturalEvolutionStrategies::BDNES.new NET.nweights_per_layer,
#   method(:fitness), :max, rseed: 10
# The random seed is fixed here to ensure a reproducible behavior
# In a real task, best using an oversized network, more iterations, and try several seeds

# NOTE: In practical applications it is best to delegate parallelization to the fitness
# function instead of computing the fitness of one individual at a time. This can be
# achieved by passing  an objective function defined on a _list_ of weight-lists, and
# setting the `parallel_fit` switch to `true`:
# nes = XNES.new NET.nweights,
#   -> (genotypes) { Parallel.map genotypes, &method(:fitness) },
#   :max, rseed: 0, parallel_fit: true


# Nothing left but to run the optimization algorithm
# Depending on the random seed (read: luck)few epochs here will suffice
50.times { nes.train }
# OK! now remember, `NET` currently holds the weights of the last evaluation
# Let's fetch the best individual found so far
best_fit, best_weights = nes.best
# Let's run them again to check they work
result = fitness best_weights
# Note if you defined a parallel fitness above you'll need instead
# result = fitness([best_weights])[0]
puts "The found network achieves a score of #{result} out of #{XOR.size} in the XOR task"
puts "Weights: #{best_weights.to_a}"
puts "Done!"
# That's it! 18 lines and you got a working neuroevolution algorithm, congrats :)
