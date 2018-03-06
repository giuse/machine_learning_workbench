
RSpec.describe MachineLearningWorkbench::NeuralNetwork do
  NN = MachineLearningWorkbench::NeuralNetwork
  netstruct = [2,2,1]

  describe NN::FeedForward do
    net = NN::FeedForward.new netstruct

    it "#initialize" do
      expect(net.struct).to eq(netstruct)
      expect(net.act_fn.call([1,2,3])).to eq(NN::FeedForward.act_fn(:sigmoid).call([1,2,3]))
    end

    it "#reset" do
      initial_state = [
        NMatrix[[0,0,1]],
        NMatrix[[0,0,1]],
        NMatrix[[0]]]
      altered_state = initial_state.collect {|m| m+1}
      net.instance_variable_set(:@state, altered_state)
      expect(net.state).not_to eq(initial_state)
      net.reset_state
      expect(net.state).to eq(initial_state)
    end

    it "#deep_reset" do
      memoized_vars = [:@layer_row_sizes, :@layer_col_sizes, :@nlayers,
        :@layer_shapes, :@nweights_per_layer, :@nweights]
      net.nweights; net.nlayers # they end up calling all methods that use memoization
      memoized_vars.each do |sym|
        expect(net.instance_variable_get(sym)).not_to be_nil
      end
      net.deep_reset
      memoized_vars.each do |sym|
        expect(net.instance_variable_get(sym)).to be_nil
      end
    end

    it "#nweights" do
      # netstruct: [2,2,1] => layer_shapes: [[2,3],[1,3]] (remember: bias!)
      expect(net.nweights).to eq(2*3 + 1*3)
    end

    it "#layer_shapes" do
      # netstruct: [2,2,1] => layer_shapes: [[2,3],[1,3]] (remember: bias!)
      expect(net.layer_row_sizes.size).to eq(net.layer_col_sizes.size)
      expect(net.layer_shapes).to eq([[2+1,2],[2+1,1]])
    end

    context "with random weights" do
      net.init_random

      it "has one output" do
        expect(net.activate([2,2]).size).to eq(1)
      end

      it "#nweights correctly counts the weights" do
        expect(net.nweights).to eq(net.weights.flatten.size)
      end
    end

    context "with loaded weights" do
      weights = net.nweights.times.collect { |n| 1.0/(n+1) } # best to avoid 1.0/0

      it "#load_weights" do
        weights_are_safe = weights.dup
        net.load_weights weights_are_safe
        expect(weights_are_safe).to eq(weights)
        expect(net.layers.collect(&:to_a).flatten).to eq(weights)
      end

      it "solves the XOR problem" do
        # [0,1].repeated_permutation(2).collect{|pair| [pair, pair.reduce(:^)]}
        xor_table = {[0,0] => 0, [1,0] => 1, [0,1] => 1, [1,1] => 0}
        net = NN::FeedForward.new([2,2,1], act_fn: :logistic)
        #              2 in + b -> 3 neur,  2 in + b -> 1 neur
        # http://stats.stackexchange.com/questions/12197/can-a-2-2-1-feedforward-neural-network-with-sigmoid-activation-functions-represe
        solution_weights = [ [[1,2],[1,2],[0,0]],  [[-1000],[850],[0]] ]
        net.load_weights solution_weights.flatten
        expect(net.weights).to eq(solution_weights)
        xor_table.each do |input, target|
          expect(net.activate(input).first.approximates? target).to be_truthy
        end
      end
    end
  end

	describe NN::Recurrent do
	  net = NN::Recurrent.new [2,2,1]
	  context "with random weights" do
	    net.init_random

	    it "#nweights and #weights correspond" do
	      expect(net.nweights).to eq(net.weights.flatten.size)
	    end

	    it "#layer_shapes" do
	      # netstruct: [2,2,1], with recurrency and biases
	      expect(net.layer_shapes).to eq([[2+2+1,2],[2+1+1,1]])
	    end

	    it "works" do
	      expect(net.activate([2,2]).size).to eq(1)
	    end

	  end
	end

end