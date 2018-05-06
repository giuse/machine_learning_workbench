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
      puts "Ignoring similarity: `simil_type: #{opts[:simil_type]}`" unless opts[:simil_type] == :dot
      puts "Ignoring ncentrs: `ncentrs: #{opts[:ncentrs]}`" if opts[:ncentrs]
      # TODO: try different epsilons to reduce the number of states
      # for example, in qbert we care what is lit and what is not, not the colors
      @equal_simil = opts.delete(:equal_simil) || 0.0
      super **opts.merge({ncentrs: 1, lrate: nil, simil_type: :dot})

      @ntrains = nil # will disable the counting
    end

    # Overloading lrate check from original VQ
    def check_lrate lrate; nil; end

    # Train on one vector:
    # - train only if the image is not already in dictionary
    # - create new centroid from the image
    # @return [Integer] index of new centroid
    def train_one vec, eps: equal_simil
      # NOTE:  novelty needs to be re-computed for each image, as after each
      # training the novelty signal changes!

# NOTE the reconstruction error here depends once more on the _color_
# this is wrong and should be taken out of the equation
# NOTE: this is fixed if I use the differences sparse coding method
      residual_img = reconstr_error(vec)
      rec_err = residual_img.mean
      return -1 if rec_err < eps
      puts "Creating centr #{ncentrs} (rec_err: #{rec_err})"
      # norm_vec = vec / NLinalg.norm(vec)
      # @centrs = centrs.concatenate norm_vec
      # @centrs = centrs.concatenate vec
      @centrs = centrs.concatenate residual_img
      # HACK: make it more general by using `code_size`
      @utility = @utility.concatenate [0] * (encoding_type == :sparse_coding_v1 ? 2 : 1)
      ncentrs
    end

  end
end
