module MachineLearningWorkbench::Compressor
  # Train-less VQ, copying new images into centroids
  # Optimized for online training.
  class CopyVQ < VectorQuantization

    attr_reader :equal_simil, :next_train

    def initialize **opts
      puts "Ignoring learning rate: `lrate: #{opts[:lrate]}`" if opts[:lrate]
      puts "Ignoring similarity: `simil_type: #{opts[:simil_type]}`" if opts[:simil_type]
      # TODO: try different epsilons to reduce the number of states
      # for example, in qbert we care what is lit and what is not, not the colors
      @equal_simil = opts.delete(:equal_simil) || 0.0
      super **opts.merge({lrate: nil, simil_type: nil})
      @ntrains << 0 # to count duplicates, images we skip the train on
      @next_train = 0 # pointer to the next centroid to train
    end

    def ntrains; @ntrains[0...-1]; end
    def ntrains_skip; @ntrains.last; end

    # Overloading lrate check from original VQ
    def check_lrate lrate; nil; end

    # Train on one vector:
    # - train only if the image is not already in dictionary
    # - find the next untrained centroid
    # - training is just overwriting it
    # @return [Integer] index of trained centroid
    def train_one vec, eps: equal_simil
      mses = centrs.map do |centr|
        ((centr-vec)**2).sum / centr.size
      end
      # BEWARE: I am currently not handling the case where we run out of centroids!
      # => Will be addressed directly by dynamic dictionary size
      # return -1 if mses.min < eps
      return -1 if mses.min < eps || next_train == ncentrs
      trg_idx = next_train
      @next_train += 1
      # require 'pry'; binding.pry if next_train == ncentrs
      puts "Overwriting centr #{next_train}"
      centrs[trg_idx] = vec
      trg_idx
    end

  end
end
