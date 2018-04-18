# frozen_string_literal: true

module MachineLearningWorkbench::Compressor
  # Incremental Dictionary Train-less VQ, creating new centroids rather than training
  # Optimized for online training.
  # TODO: as the deadline grows nigh, the hacks grow foul. Refactor all VQs together.
  class IncrDictVQ < VectorQuantization

    attr_reader :equal_simil
    undef :ntrains # centroids are not trained

    def initialize **opts
      puts "Ignoring learning rate: `lrate: #{opts[:lrate]}`" if opts[:lrate]
      puts "Ignoring similarity: `simil_type: #{opts[:simil_type]}`" if opts[:simil_type]
      puts "Ignoring ncentrs: `ncentrs: #{opts[:ncentrs]}`" if opts[:ncentrs]
      # TODO: try different epsilons to reduce the number of states
      # for example, in qbert we care what is lit and what is not, not the colors
      @equal_simil = opts.delete(:equal_simil) || 0.0
      super **opts.merge({ncentrs: 1, lrate: nil, simil_type: nil})
      @ntrains = nil # will disable the counting
    end

    # Overloading lrate check from original VQ
    def check_lrate lrate; nil; end

    # Train on one vector:
    # - train only if the image is not already in dictionary
    # - create new centroid from the image
    # @return [Integer] index of new centroid
    def train_one vec, eps: equal_simil
      mses = centrs.map do |centr|
        ((centr-vec)**2).sum / centr.size # uhm get rid of division maybe? squares?
      end
      # skip training if the centr with smallest mse (most similar) has less than eps error (equal)
      # TODO: maintain an average somewhere, make eps dynamic
      return if mses.min < eps
      puts "Creating centr #{ncentrs}"
      centrs << vec
      @ncentrs.tap{ @ncentrs += 1}
    end

  end
end
