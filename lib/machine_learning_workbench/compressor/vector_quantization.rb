module MachineLearningWorkbench::Compressor
  class VectorQuantization
    attr_reader :ncentrs, :centrs, :dims, :vrange, :dtype, :lrate, :rng
    Verification = MachineLearningWorkbench::Tools::Verification

    def initialize ncentrs:, dims:, vrange:, dtype:, lrate:, rseed: Random.new_seed
      @rng = Random.new rseed
      @ncentrs = ncentrs
      @dtype = dtype
      @dims = Array(dims)
      @lrate = lrate
      @vrange = case vrange
        when Array
          raise ArgumentError, "vrange size not 2: #{vrange}" unless vrange.size == 2
          vrange.map &method(:Float)
        when Range then [vrange.first, vrange.last].map &method(:Float)
        else raise ArgumentError, "vrange: unrecognized type: #{vrange.class}"
      end
      @centrs = ncentrs.times.map { new_centr }
    end

    # Creates a new (random) centroid
    def new_centr
      NMatrix.new(dims, dtype: dtype) { rng.rand Range.new *vrange }
    end

    # Computes similarities between image and all centroids
    def similarities img
      raise NotImplementedError if img.shape.size > 1
      # centrs.map { |c| c.dot(img).first }
      require 'parallel'
      Parallel.map(centrs) { |c| c.dot(img).first }
    end
    # The list of similarities also constitutes the encoding of the image
    alias encode similarities

    # Returns index and similitude of most similar centroid to image
    def most_similar_centr img
      simils = similarities img
      max_simil = simils.max
      max_idx = simils.index max_simil
      [max_idx, max_simil]
    end

    # Reconstruct image as its most similar centroid
    def reconstruction img
      centrs[most_similar_centr(img).first]
    end

    # Per-pixel errors in reconstructing image
    def reconstr_error img
      reconstruction(img) - img
    end

    # Train on one image
    def train_one img, simils: nil
      trg_idx, _simil = simils || most_similar_centr(img)
      centrs[trg_idx] = centrs[trg_idx] * (1-lrate) + img * lrate
      Verification.in_range! centrs[trg_idx], vrange
      centrs[trg_idx]
    end

    # Train on image list
    def train img_lst, debug: false
      # Two ways here:
      # - Batch: canonical, centrs updated with each img
      # - Parallel: could be parallel either on simils or on training (?)
      # Unsure on the correctness of either Parallel, let's stick with Batch
      img_lst.each { |img| train_one img; print '.' if debug }
    end
  end
end
