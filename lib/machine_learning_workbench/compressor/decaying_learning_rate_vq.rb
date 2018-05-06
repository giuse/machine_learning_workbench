# frozen_string_literal: true

module MachineLearningWorkbench::Compressor
  # VQ with per-centroid decaying learning rates.
  # Optimized for online training.
  class DecayingLearningRateVQ < VectorQuantization

    attr_reader :lrate_min, :lrate_min_den, :decay_rate

    def initialize **opts
      puts "Ignoring learning rate: `lrate: #{opts[:lrate]}`" if opts[:lrate]
      @lrate_min = opts.delete(:lrate_min) || 0.001
      @lrate_min_den = opts.delete(:lrate_min_den) || 1
      @decay_rate = opts.delete(:decay_rate) || 1
      super **opts.merge({lrate: nil})
    end

    # Overloading lrate check from original VQ
    def check_lrate lrate; nil; end

    # Decaying per-centroid learning rate.
    # @param centr_idx [Integer] index of the centroid
    # @param lower_bound [Float] minimum learning rate
    # @note nicely overloads the `attr_reader` of parent class
    def lrate centr_idx, min_den: lrate_min_den, lower_bound: lrate_min, decay: decay_rate
      [1.0/(ntrains[centr_idx]*decay+min_den), lower_bound].max
      .tap { |l| puts "centr: #{centr_idx}, ntrains: #{ntrains[centr_idx]}, lrate: #{l}" }
    end

    # Train on one vector
    # @return [Integer] index of trained centroid
    def train_one vec, eps: nil
      # NOTE: ignores epsilon if passed
      trg_idx, _simil = most_similar_centr(vec)
      centrs[trg_idx, true] = centrs[trg_idx, true] * (1-lrate(trg_idx)) + vec * lrate(trg_idx)
      trg_idx
    end

  end
end
