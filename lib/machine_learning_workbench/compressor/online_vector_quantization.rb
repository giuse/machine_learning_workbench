module MachineLearningWorkbench::Compressor
  # Online Vector Quantization: VQ with per-centroid decaying learning rates.
  # Optimized for online training.
  class OnlineVectorQuantization < VectorQuantization

    attr_reader :min_lrate

    def initialize min_lrate: 0.01, **opts
      super **opts.merge({lrate: nil})
      @min_lrate = min_lrate
    end

    # Overloading lrate check from original VQ
    def check_lrate lrate; nil; end

    # Decaying per-centroid learning rate.
    # @param centr_idx [Integer] index of the centroid
    # @param lower_bound [Float] minimum learning rate
    # @note nicely overloads the `attr_reader` of parent class
    def lrate centr_idx, lower_bound: min_lrate
      [1/ntrains[centr_idx], lower_bound].max
    end

    # Train on one vector
    # @return [Integer] index of trained centroid
    def train_one vec
      trg_idx, _simil = most_similar_centr(vec)
      centrs[trg_idx] = centrs[trg_idx] * (1-lrate(trg_idx)) + vec * lrate(trg_idx)
      trg_idx
    end

  end
end
