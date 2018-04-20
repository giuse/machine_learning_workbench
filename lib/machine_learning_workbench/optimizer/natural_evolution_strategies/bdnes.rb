# frozen_string_literal: true

module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Block-Diagonal Natural Evolution Strategies
  class BDNES < Base

    MAX_RSEED = 10**Random.new_seed.size # block random seeds to be on the same range as `Random.new_seed`

    attr_reader :ndims_lst, :blocks, :popsize, :parallel_update
    undef :ndims # only `ndims_lst` here

    # Initialize a list of XNES, one for each block
    # see class `Base` for the description of the rest of the arguments.
    # @param ndims_lst [Array<Integer>] list of sizes for each block in the block-diagonal
    #    matrix. Note: entire (reconstructed) individuals will be passed to the `obj_fn`
    #    regardless of the division here described.
    # @param init_opts [Hash] the rest of the options will be passed directly to XNES
    # @parellel_update [bool] whether to parallelize block updates
    def initialize ndims_lst, obj_fn, opt_type, parallel_fit: false, rseed: nil, parallel_update: false, **init_opts
      # mu_init: 0, sigma_init: 1
      # init_opts = {rseed: rseed, mu_init: mu_init, sigma_init: sigma_init}
      # TODO: accept list of `mu_init`s and `sigma_init`s
      @ndims_lst, @obj_fn, @opt_type, @parallel_fit = ndims_lst, obj_fn, opt_type, parallel_fit
      block_fit = -> (*args) { raise "Should never be called" }
      # the BD-NES seed should ensure deterministic reproducibility
      # but each block should have a different seed
      # puts "BD-NES rseed: #{s}"  # currently disabled
      @rng = Random.new rseed || Random.new_seed
      @blocks = ndims_lst.map do |ndims|
        b_rseed = rng.rand MAX_RSEED
        XNES.new ndims, block_fit, opt_type, rseed: b_rseed, **init_opts
      end
      # Need `popsize` to be the same for all blocks, to make complete individuals
      @popsize = blocks.map(&:popsize).max
      blocks.each { |xnes| xnes.instance_variable_set :@popsize, popsize }

      @best = [(opt_type==:max ? -1 : 1) * Float::INFINITY, nil]
      @last_fits = []
      @parallel_update = parallel_update
      require 'parallel' if parallel_update
    end

    def sorted_inds_lst
      # Build samples and inds from the list of blocks
      samples_lst, inds_lst = blocks.map do |xnes|
        samples = xnes.standard_normal_samples
        inds = xnes.move_inds(samples)
        [samples.to_a, inds]
      end.transpose

      # Join the individuals for evaluation
      full_inds = inds_lst.reduce { |mem, var| mem.concatenate var, axis: 1 }
      # Need to fix sample dimensions for sorting
      # - current dims: nblocks x ninds x [block sizes]
      # - for sorting: ninds x nblocks x [block sizes]
      full_samples = samples_lst.transpose

      # Evaluate fitness of complete individuals
      fits = parallel_fit ? obj_fn.call(full_inds) : full_inds.map(&obj_fn)
      # Quick cure for NaN fitnesses
      fits.map { |x| x.nan? ? (opt_type==:max ? -1 : 1) * Float::INFINITY : x }
      @last_fits = fits # allows checking for stagnation

      # Sort inds based on fit and opt_type, save best
      # sorted = [fits, full_inds, full_samples].transpose.sort_by(&:first)
      # sorted.reverse! if opt_type==:min
      # this_best = sorted.last.take(2)
      # opt_cmp_fn = opt_type==:min ? :< : :>
      # @best = this_best if this_best.first.send(opt_cmp_fn, best.first)
      # sorted_samples = sorted.map(&:last)
      sort_idxs = fits.sort_index
      sort_idxs = sort_idxs.reverse if opt_type == :min
      this_best = [fits[sort_idxs[-1]], full_inds[sort_idxs[-1], true]]
      opt_cmp_fn = opt_type==:min ? :< : :>
      @best = this_best if this_best.first.send(opt_cmp_fn, best.first)
      sorted_samples = full_samples.values_at *sort_idxs

      # Need to bring back sample dimensions for each block
      # - current dims: ninds x nblocks x [block sizes]
      # - target blocks list: nblocks x ninds x [block sizes]
      block_samples = sorted_samples.transpose

      # then back to NArray for usage in training
      block_samples.map &:to_na
    end

    # duck-type the interface: [:train, :mu, :convergence, :save, :load]

    # TODO: refactor DRY
    def train picks: sorted_inds_lst
      if parallel_update
        # Parallel.each(blocks.zip(picks)) do |xnes, s_inds|
        #   xnes.train picks: s_inds
        # end
        # Actually it's not this simple.
        # Forks do not act on the parent, so I need to send back updated mu and sigma
        # Luckily we have `NES#save` and `NES#load` at the ready
        # Next: need to implement `#marshal_dump` and `#marshal_load` in `Base`
        # Actually using `Cumo` rather than `Parallel` may avoid marshaling altogether
        raise NotImplementedError, "Should dump and load each instance"
      else
        blocks.zip(picks).each do |xnes, s_inds|
          xnes.train picks: s_inds
        end
      end
    end

    def mu
      blocks.map(&:mu).reduce { |mem, var| mem.concatenate var, axis: 1 }
    end

    def sigma
      raise NotImplementedError, "need to write a concatenation like for mu here"
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
