
module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Block-Diagonal Natural Evolution Strategies
  class BDNES < Base

    MAX_RSEED = 10**Random.new_seed.size # same range as Random.new_seed

    attr_reader :ndims_lst, :obj_fn, :opt_type, :parallel_fit, :blocks, :popsize, :rng,
      :best, :last_fits

    # Initialize a list of XNES, one for each block
    # see class `Base` for the description of the rest of the arguments.
    # @param ndims_lst [Array<Integer>] list of sizes for each block in the block-diagonal
    #    matrix. Note: entire (reconstructed) individuals will be passed to the `obj_fn`
    #    regardless of the division here described.
    # @param init_opts [Hash] the rest of the options will be passed directly to XNES
    def initialize ndims_lst, obj_fn, opt_type, parallel_fit: false, rseed: nil, **init_opts
      # mu_init: 0, sigma_init: 1
      # init_opts = {rseed: rseed, mu_init: mu_init, sigma_init: sigma_init}
      # TODO: accept list of `mu_init`s and `sigma_init`s
      @ndims_lst, @obj_fn, @opt_type, @parallel_fit = ndims_lst, obj_fn, opt_type, parallel_fit
      block_fit = -> (*args) { raise "Should never be called" }
      # the BD-NES seed should ensure deterministic reproducibility
      # but each block should have a different seed
      rseed ||= Random.new_seed
      # puts "BD-NES rseed: #{s}"  # currently disabled
      @rng = Random.new rseed
      @blocks = ndims_lst.map do |ndims|
        b_rseed = rng.rand MAX_RSEED
        XNES.new ndims, block_fit, opt_type, rseed: b_rseed, **init_opts
      end
      # Need `popsize` to be the same for all blocks, to make complete individuals
      @popsize = blocks.map(&:popsize).max
      blocks.each { |xnes| xnes.instance_variable_set :@popsize, popsize }

      @best = [(opt_type==:max ? -1 : 1) * Float::INFINITY, nil]
      @last_fits = []
    end

    def sorted_inds_lst
      # Build samples and inds from the list of blocks
      samples_lst, inds_lst = blocks.map do |xnes|
        samples = xnes.standard_normal_samples
        inds = xnes.move_inds(samples)
        [samples.to_a, inds]
      end.transpose

      # Join the individuals for evaluation
      full_inds = inds_lst.reduce(&:hconcat).to_a
      # Need to fix samples dimensions for sorting
      # - current dims: nblocks x ninds x [block sizes]
      # - for sorting: ninds x nblocks x [block sizes]
      full_samples = samples_lst.transpose

      # Evaluate fitness of complete individuals
      fits = parallel_fit ? obj_fn.call(full_inds) : full_inds.map(&obj_fn)
      # Quick cure for NaN fitnesses
      fits.map! { |x| x.nan? ? (opt_type==:max ? -1 : 1) * Float::INFINITY : x }
      @last_fits = fits # allows checking for stagnation

      # Sort inds based on fit and opt_type, save best
      sorted = [fits, full_inds, full_samples].transpose.sort_by(&:first)
      sorted.reverse! if opt_type==:min
      this_best = sorted.last.take(2)
      opt_cmp_fn = opt_type==:min ? :< : :>
      @best = this_best if this_best.first.send(opt_cmp_fn, best.first)
      sorted_samples = sorted.map(&:last)

      # Need to bring back sample dimensions for each block
      # - current dims: ninds x nblocks x [block sizes]
      # - target blocks list: nblocks x ninds x [block sizes]
      block_samples = sorted_samples.transpose

      # then back to NMatrix for usage in training
      block_samples.map { |sample| NMatrix[*sample, dtype: :float64] }
    end

    # duck-type the interface: [:train, :mu, :convergence, :save, :load]

    def train picks: sorted_inds_lst
      blocks.zip(sorted_inds_lst).each do |xnes, s_inds|
        xnes.train picks: s_inds
      end
    end

    def mu
      blocks.map(&:mu).reduce(&:hconcat)
    end

    def convergence
      blocks.map(&:convergence).reduce(:+)
    end

    def save
      blocks.map &:save
    end

    def load data
      fit = -> (*args) { raise "Should never be called" }
      @blocks = data.map do |block_data|
        ndims = block_data.first.size
        XNES.new(ndims, fit, opt_type).tap do |nes|
          nes.load block_data
        end
      end
    end
  end
end