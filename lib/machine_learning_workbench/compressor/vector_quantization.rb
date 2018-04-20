# frozen_string_literal: true

module MachineLearningWorkbench::Compressor

  # Standard Vector Quantization
  class VectorQuantization
    attr_reader :ncentrs, :centrs, :dims, :vrange, :init_centr_vrange, :lrate,
      :simil_type, :encoding_type, :rng, :ntrains, :utility, :ncodes
    attr_writer :utility, :ncodes # allows access from outside

    def initialize ncentrs:, dims:, vrange:, lrate:, simil_type: nil, encoding_type: nil, init_centr_vrange: nil, rseed: Random.new_seed

      @rng = Random.new rseed # TODO: RNG CURRENTLY NOT USED!!

      @ncentrs = ncentrs
      @dims = Array(dims)
      check_lrate lrate # hack: so that we can overload it in dlr_vq
      @lrate = lrate
      @simil_type = simil_type || :dot
      @encoding_type = encoding_type || :norm_ensemble
      @init_centr_vrange ||= vrange
      @vrange = case vrange
        when Array
          raise ArgumentError, "vrange size not 2: #{vrange}" unless vrange.size == 2
          vrange.map &method(:Float)
        when Range
          [vrange.first, vrange.last].map &method(:Float)
        else raise ArgumentError, "vrange: unrecognized type: #{vrange.class}"
      end
      init_centrs
      @ntrains = [0]*ncentrs # useful to understand what happens
      @utility = NArray.zeros [ncentrs] # trace how 'useful' are centroids to encodings
      @ncodes = 0
    end

    # Verify lrate to be present and withing unit bounds
    # As a separate method only so it can be overloaded in `DecayingLearningRateVQ`
    def check_lrate lrate
      raise ArgumentError, "Pass a `lrate` between 0 and 1" unless lrate&.between?(0,1)
    end

    # Initializes a list of centroids
    def init_centrs nc: ncentrs, base: nil, proport: nil
      @centrs = nc.times.map { new_centr base, proport }
    end

    # Creates a new (random) centroid
    # If a base is passed, this is meshed with the random centroid.
    # This is done to facilitate distributing the training across centroids.
    # TODO: USE RNG HERE!!
    def new_centr base=nil, proport=nil
      raise ArgumentError, "Either both or none" if base.nil? ^ proport.nil?
      # require 'pry'; binding.pry if base.nil? ^ proport.nil?
      ret = NArray.new(*dims).rand(*init_centr_vrange)
      ret = ret * (1-proport) + base * proport if base&&proport
      ret
    end

    SIMIL = {
      dot: -> (centr, vec) { centr.dot(vec) },
      mse: -> (centr, vec) { -((centr-vec)**2).sum / centr.size }
    }

    # Computes similarities between vector and all centroids
    def similarities vec, type: simil_type
      raise NotImplementedError if vec.shape.size > 1
      simil_fn = SIMIL[type] || raise(ArgumentError, "Unrecognized simil #{type}")
      NArray[*centrs.map { |centr| simil_fn.call centr, vec }]
      # require 'parallel'
      # NArray[*Parallel.map(centrs) { |c| c.dot(vec).first }]
    end

    # Encode a vector
    # tracks utility of centroids based on how much they contribute to encoding
    # TODO: `encode = Encodings.const_get(type)` in initialize`
    # NOTE: hashes of lambdas or modules cannot access ncodes and utility
    def encode vec, type: encoding_type
      simils = similarities vec
      case type
      when :most_similar
        code = simils.max_index
        @ncodes += 1
        @utility[code] += 1
        code
      when :most_similar_ary
        code = simils.new_zeros
        code[simils.max_index] = 1
        @ncodes += 1
        @utility += code
        code
      when :ensemble
        code = simils
        tot = simils.sum
        tot = 1 if tot < 1e-5  # HACK: avoid division by zero
        contrib = code / tot
        @ncodes += 1
        @utility += (contrib - utility) / ncodes # cumulative moving average
        code
      when :norm_ensemble
        tot = simils.sum
        tot = 1 if tot < 1e-5  # HACK: avoid division by zero
        code = simils / tot
        @ncodes += 1
        @utility += (code - utility) / ncodes # cumulative moving average
        code
      when :sparse_coding
        raise NotImplementedError, "do this next"


        @ncodes += 1
        @utility += (code - utility) / ncodes # cumulative moving average
        code
      else raise ArgumentError, "Unrecognized encode #{type}"
      end
    end

    # Reconstruct vector from its code (encoding)
    def reconstruction code, type: encoding_type
      case type
      when :most_similar
        centrs[code]
      when :most_similar_ary
        centrs[code.eq(1).where[0]]
      when :ensemble
        tot = code.reduce :+
        centrs.zip(code).map { |centr, contr| centr*contr/tot }.reduce :+
      when :norm_ensemble
        centrs.zip(code).map { |centr, contr| centr*contr }.reduce :+
      when :sparse_coding
        raise NotImplementedError, "do this next"
      else raise ArgumentError, "unrecognized reconstruction type: #{type}"
      end
    end

    # Returns index and similitude of most similar centroid to vector
    # @return [Array<Integer, Float>] the index of the most similar centroid,
    #   followed by the corresponding similarity
    def most_similar_centr vec
      simils = similarities vec
      max_idx = simils.max_index
      [max_idx, simils[max_idx]]
    end

    # Per-pixel errors in reconstructing vector
    # @return [NArray] residuals
    def reconstr_error vec, code: nil, type: encoding_type
      code ||= encode vec, type: type
      (vec - reconstruction(code, type: type)).abs.sum
    end

    # Train on one vector
    # @return [Integer] index of trained centroid
    def train_one vec
      trg_idx, _simil = most_similar_centr(vec)
      # note: uhm that actually looks like a dot product... maybe faster?
      #   `[c[i], vec].dot([1-lrate, lrate])`
      centrs[trg_idx] = centrs[trg_idx] * (1-lrate) + vec * lrate
      trg_idx
    end

    # Train on vector list
    def train vec_lst, debug: false
      # Two ways here:
      # - Batch: canonical, centrs updated with each vec
      # - Parallel: could be parallel either on simils or on training (?)
      # Unsure on the correctness of either Parallel, let's stick with Batch
      vec_lst.each_with_index do |vec, i|
        trained_idx = train_one vec
        print '.' if debug
        @ntrains[trained_idx] += 1 if @ntrains
      end
    end
  end
end
