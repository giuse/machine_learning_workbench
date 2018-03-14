module MachineLearningWorkbench::Compressor
  # Online Vector Quantization: VQ with per-centroid decaying learning rates.
  # Optimized for online training.
  class OnlineVectorQuantization < VectorQuantization

    attr_reader :min_lrate, :ntrains

    def initialize min_lrate: 0.01, **opts
      super **opts.merge({lrate: nil})
      @min_lrate = min_lrate
      @ntrains = [0]*ncentrs
    end

    # Decaying per-centroid learning rate.
    # @param centr_idx [Integer] index of the centroid
    # @param lower_bound [Float] minimum learning rate
    def lrate centr_idx, lower_bound: min_lrate
      [1/ntrains[centr_idx], lower_bound].max
    end

    # Train on one image
    # @return [Integer] index of trained centroid
    def train_one *args, **opts
      super.tap { |trg_idx| ntrains[trg_idx] += 1 }
    end
  end
end
