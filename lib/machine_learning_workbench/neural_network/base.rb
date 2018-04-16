# frozen_string_literal: true

module MachineLearningWorkbench::NeuralNetwork
  # Neural Network base class
  class Base

    # @!attribute [r] layers
    #   List of matrices, each being the weights
    #   connecting a layer's inputs (rows) to a layer's neurons (columns),
    #   hence its shape is `[ninputs, nneurs]`
    #   @return [Array<NArray>] list of weight matrices, each uniquely describing a layer
    #   TODO: return a NArray after the usage of `#map` is figured out
    # @!attribute [r] state
    #   It's a list of one-dimensional matrices, each an input to a layer, plus the output layer's output. The first element is the input to the first layer of the network, which is composed of the network's input, possibly the first layer's activation on the last input (recursion), and a bias (fixed `1`). The second to but-last entries follow the same structure, but with the previous layer's output in place of the network's input. The last entry is the activation of the output layer, without additions since it's not used as an input by anyone.
    #   TODO: return a NArray after the usage of `#map` is figured out
    #   @return [Array<NArray>] current state of the network.
    # @!attribute [r] act_fn
    #   activation function, common to all neurons (for now)
    #   @return [#call] activation function
    # @!attribute [r] struct
    #   list of number of (inputs or) neurons in each layer
    #   @return [Array<Integer>] structure of the network
    attr_reader :layers, :state, :act_fn, :act_fn_name, :struct


    ## Initialization

    # @param struct [Array<Integer>] list of layer sizes
    # @param act_fn [Symbol] choice of activation function for the neurons
    def initialize struct, act_fn: nil
      @struct = struct
      @act_fn_name = act_fn || :sigmoid
      @act_fn = send(act_fn_name)
      # @state holds both inputs, possibly recurrency, and bias
      # it is a complete input for the next layer, hence size from layer sizes
      @state = layer_row_sizes.collect do |size|
        NArray.zeros [1, size]
      end
      # to this, append a matrix to hold the final network output
      @state.push NArray.zeros [1, nneurs(-1)]
      reset_state
    end

    # Reset the network to the initial state
    def reset_state
      state.each do |s|
        s.fill 0           # reset state to zero
        s[-1] = 1        # add bias
      end
      state[-1][-1] = 0  # last layer has no bias
    end

    # Initialize the network with random weights
    def init_random
      # Reusing `#load_weights` instead helps catching bugs
      load_weights NArray.new(nweights).rand(-1,1)
    end

    ## Weight utilities

    # Resets memoization: needed to play with structure modification
    def deep_reset
      # reset memoization
      [:@layer_row_sizes, :@layer_col_sizes, :@nlayers, :@layer_shapes,
       :@nweights_per_layer, :@nweights].each do |sym|
         instance_variable_set sym, nil
      end
      reset_state
    end

    # Total weights in the network
    # @return [Integer] total number of weights
    def nweights
      @nweights ||= nweights_per_layer.reduce(:+)
    end

    # List of per-layer number of weights
    # @return [Array<Integer>] list of weights per each layer
    def nweights_per_layer
      @nweights_per_layer ||= layer_shapes.collect { |shape| shape.reduce(:*) }
    end

    # Count the layers. This is a computation helper, and for this implementation
    # the inputs are considered as if a layer like the others.
    # @return [Integer] number of layers
    def nlayers
      @nlayers ||= layer_shapes.size
    end

    # Returns the weight matrix
    # @return [Array<NArray>] list of NArray matrices of weights (one per layer).
    def weights
      layers
    end

    # Number of neurons per layer. Although this implementation includes inputs
    # in the layer counts, this methods correctly ignores the input as not having
    # neurons.
    # @return [Array] list of neurons per each (proper) layer (i.e. no inputs)
    def layer_col_sizes
      @layer_col_sizes ||= struct.drop(1)
    end

    # define #layer_row_sizes in child class: number of inputs per layer

    # Shapes for the weight matrices, each corresponding to a layer
    # @return [Array<Array[Integer, Integer]>] Weight matrix shapes
    def layer_shapes
      @layer_shapes ||= layer_row_sizes.zip layer_col_sizes
    end

    # Count the neurons in a particular layer or in the whole network.
    # @param nlay [Integer, nil] the layer of interest, 1-indexed.
    #   `0` will return the number of inputs.
    #   `nil` will compute the total neurons in the network.
    # @return [Integer] the number of neurons in a given layer, or in all network, or the number of inputs
    def nneurs nlay=nil
      nlay.nil? ? struct.reduce(:+) : struct[nlay]
    end

    # Loads a plain list of weights into the weight matrices (one per layer).
    # Preserves order. Reuses allocated memory if available.
    # @input weights [Array<Float>] weights to load
    # @return [true] always true. If something's wrong it simply fails, and if
    #   all goes well there's nothing to return but a confirmation to the caller.
    def load_weights weights
      raise ArgumentError unless weights.size == nweights
      weights = weights.to_na unless weights.kind_of? NArray
      from = 0
      @layers = layer_shapes.collect do |shape|
        to = from + shape.reduce(:*)
        lay_w = weights[from...to].reshape *shape
        from = to
        lay_w
      end
      reset_state
      return true
    end


    ## Activation

    # Activate the network on a given input
    # @param input [Array<Float>] the given input
    # @return [Array] the activation of the output layer
    def activate input
      raise ArgumentError unless input.size == struct.first
      # load input in first state
      state[0][0...struct.first] = input
      # activate layers in sequence
      nlayers.times.each do |i|
        act = activate_layer i
        state[i+1][0...act.size] = act
      end
      return out
    end

    # Extract and convert the output layer's activation
    # @return [NArray] the activation of the output layer
    def out
      state.last.flatten
    end

    ## Activation functions

    # Traditional sigmoid (logistic) with variable steepness
    def sigmoid k=1
      # k is steepness:  0<k<1 is flatter, 1<k is flatter
      # flatter makes activation less sensitive, better with large number of inputs
      -> (vec) { 1.0 / (NMath.exp(-k * vec) + 1.0) }
    end
    alias logistic sigmoid

    # LeCun hyperbolic activation
    # @see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf Section 4.4
    def lecun_hyperbolic
      -> (vec) { 1.7159 * NMath.tanh(2.0*vec/3.0) + 1e-3*vec }
    end

    # Rectified Linear Unit (ReLU)
    def relu
      -> (vec) { (vec>0).all? && vec || vec.class.zeros(vec.shape) }
    end


    # @!method interface_methods
    # Declaring interface methods - implement in child class!
    [:layer_row_sizes, :activate_layer].each do |sym|
      define_method sym do
        raise NotImplementedError, "Implement ##{sym} in child class!"
      end
    end
  end
end
