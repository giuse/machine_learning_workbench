module MachineLearningWorkbench::Compressor

  # Standard Vector Quantization
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
      # TODO: this is too slow, find another way to use the rng
      # NMatrix.new(dims, dtype: dtype) { rng.rand Range.new *vrange }
      NMatrix.random dims, dtype: dtype
    end

    # Computes similarities between vector and all centroids
    def similarities vec
      raise NotImplementedError if vec.shape.size > 1
      centrs.map { |c| c.dot(vec).first }
      # require 'parallel'
      # Parallel.map(centrs) { |c| c.dot(vec).first }
    end

    # Encode a vector
    def encode vec, type: :most_similar
      simils = similarities vec
      case type
      when :most_similar
        simils.index simils.max
      when :ensemble
        simils
      when :ensemble_norm
        tot = simils.reduce(:+)
        simils.map { |s| s/tot }
      else raise ArgumentError, "unrecognized encode type: #{type}"
      end
    end

    # Reconstruct vector from its code (encoding)
    def reconstruction code, type: :most_similar
      case type
      when :most_similar
        centrs[code]
      when :ensemble
        tot = code.reduce :+
        centrs.zip(code).map { |centr, contr| centr*contr/tot }.reduce :+
      when :ensemble_norm
        centrs.zip(code).map { |centr, contr| centr*contr }.reduce :+
      else raise ArgumentError, "unrecognized reconstruction type: #{type}"
      end
    end

    # Returns index and similitude of most similar centroid to vector
    def most_similar_centr vec
      simils = similarities vec
      max_simil = simils.max
      max_idx = simils.index max_simil
      [max_idx, max_simil]
    end

    # Per-pixel errors in reconstructing vector
    def reconstr_error vec
      reconstruction(vec) - vec
    end

    # Train on one vector
    # @param vec [NMatrix]
    # @return [Integer] index of trained centroid
    def train_one vec, simils: nil
      trg_idx, _simil = simils || most_similar_centr(vec)
      centrs[trg_idx] = centrs[trg_idx] * (1-lrate) + vec * lrate
      Verification.in_range! centrs[trg_idx], vrange
      trg_idx
    end

    # Train on vector list
    def train vec_lst, debug: false
      # Two ways here:
      # - Batch: canonical, centrs updated with each vec
      # - Parallel: could be parallel either on simils or on training (?)
      # Unsure on the correctness of either Parallel, let's stick with Batch
      vec_lst.each { |vec| train_one vec; print '.' if debug }
    end
  end
end
