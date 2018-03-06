
module MachineLearningWorkbench::NeuralNetwork
  # Feed Forward Neural Network
  class FeedForward < Base

    # Calculate the size of each row in a layer's weight matrix.
    # Includes inputs (or previous-layer activations) and bias.
    # @return [Array<Integer>] per-layer row sizes
    def layer_row_sizes
      @layer_row_sizes ||= struct.each_cons(2).collect {|prev, _curr| prev+1}
    end

    # Activates a layer of the network
    # @param i [Integer] the layer to activate, zero-indexed
    def activate_layer i
      act_fn.call( state[i].dot layers[i] )
    end

  end
end