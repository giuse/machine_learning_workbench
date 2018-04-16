# frozen_string_literal: true

module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Natural Evolution Strategies base class
  class Base
    attr_reader :ndims, :mu, :sigma, :opt_type, :obj_fn, :parallel_fit, :eye, :rng, :last_fits, :best, :rescale_popsize, :rescale_lrate

    # NES object initialization
    # @param ndims [Integer] number of parameters to optimize
    # @param obj_fn [#call] any object defining a #call method (Proc, lambda, custom class)
    # @param opt_type [:min, :max] select minimization / maximization of obj_fn
    # @param rseed [Integer] allow for deterministic execution on rseed provided
    # @param mu_init [Numeric] values to initalize the distribution's mean
    # @param sigma_init [Numeric] values to initialize the distribution's covariance
    # @param parallel_fit [boolean] whether the `obj_fn` should be passed all the
    #   individuals together. In the canonical case the fitness function always scores a
    #   single individual; in practical cases though it is easier to delegate the scoring
    #   parallelization to the external fitness function. Turning this to `true` will make
    #   the algorithm pass _an Array_ of individuals to the fitness function, rather than
    #   a single instance.
    # @param rescale_popsize [Float] scaling for the default population size
    # @param rescale_lrate [Float] scaling for the default learning rate
    def initialize ndims, obj_fn, opt_type, rseed: nil, mu_init: 0, sigma_init: 1, parallel_fit: false, rescale_popsize: 1, rescale_lrate: 1
      raise ArgumentError unless [:min, :max].include? opt_type
      raise ArgumentError unless obj_fn.respond_to? :call
      @ndims, @opt_type, @obj_fn, @parallel_fit = ndims, opt_type, obj_fn, parallel_fit
      @rescale_popsize, @rescale_lrate = rescale_popsize, rescale_lrate
      @eye = NArray.eye(ndims)
      rseed ||= Random.new_seed
      # puts "NES rseed: #{s}"  # currently disabled
      @rng = Random.new rseed
      @best = [(opt_type==:max ? -1 : 1) * Float::INFINITY, nil]
      @last_fits = []
      initialize_distribution mu_init: mu_init, sigma_init: sigma_init
    end

    # Box-Muller transform: generates standard (unit) normal distribution samples
    # @return [Float] a single sample from a standard normal distribution
    # @note Xumo::NArray implements this but no random seed selection yet
    def standard_normal_sample
      rho = Math.sqrt(-2.0 * Math.log(rng.rand))
      theta = 2 * Math::PI * rng.rand
      tfn = rng.rand > 0.5 ? :cos : :sin
      rho * Math.send(tfn, theta)
    end

    # Memoized automatic magic numbers
    # NOTE: Doubling popsize and halving lrate often helps
    def utils;   @utilities ||= cmaes_utilities end
    # (see #utils)
    def popsize; @popsize   ||= cmaes_popsize * rescale_popsize end
    # (see #utils)
    def lrate;   @lrate     ||= cmaes_lrate * rescale_lrate end

    # Magic numbers from CMA-ES (TODO: add proper citation)
    # @return [NArray] scale-invariant utilities
    def cmaes_utilities
      # Algorithm equations are meant for fitness maximization
      # Match utilities with individuals sorted by INCREASING fitness
      log_range = (1..popsize).collect do |v|
        [0, Math.log(popsize.to_f/2 - 1) - Math.log(v)].max
      end
      total = log_range.reduce(:+)
      buf = 1.0/popsize
      vals = log_range.collect { |v| v / total - buf }.reverse
      NArray[vals]
    end

    # (see #cmaes_utilities)
    # @return [Float] learning rate lower bound
    def cmaes_lrate
      (3+Math.log(ndims)) / (5*Math.sqrt(ndims))
    end

    # (see #cmaes_utilities)
    # @return [Integer] population size lower bound
    def cmaes_popsize
      [5, 4 + (3*Math.log(ndims)).floor].max
    end

    # Samples a standard normal distribution to construct a NArray of
    #   popsize multivariate samples of length ndims
    # @return [NArray] standard normal samples
    # @note Xumo::NArray implements this but no random seed selection yet
    def standard_normal_samples
      NArray.zeros([popsize, ndims]).tap do |ret|
        ret.each_with_index { |_,*i| ret[*i] = standard_normal_sample }
      end
    end

    # Move standard normal samples to current distribution
    # @return [NArray] individuals
    def move_inds inds
      # TODO: can we reduce the transpositions?

      # multi_mu = NMatrix[*inds.rows.times.collect {mu.to_a}, dtype: dtype].transpose
      # (multi_mu + sigma.dot(inds.transpose)).transpose

      mu_tile = mu.tile(inds.shape.first, 1).transpose
      (mu_tile + sigma.dot(inds.transpose)).transpose
    end

    # Sorted individuals
    # NOTE: Algorithm equations are meant for fitness maximization. Utilities need to be
    # matched with individuals sorted by INCREASING fitness. Then reverse order for minimization.
    # @return standard normal samples sorted by the respective individuals' fitnesses
    def sorted_inds
      # Xumo::NArray implements the Box-Muller, but no random seed (yet)
      samples = standard_normal_samples
      # samples = NArray.new([popsize, ndims]).rand_norm(0,1)
      inds = move_inds(samples)
      fits = parallel_fit ? obj_fn.call(inds) : inds.map(&obj_fn)
      # Quick cure for NaN fitnesses
      fits.map { |x| x.nan? ? (opt_type==:max ? -1 : 1) * Float::INFINITY : x }
      @last_fits = fits # allows checking for stagnation

      # sorted = [fits.to_a, inds, samples.to_a].transpose.sort_by(&:first)
      # sorted.reverse! if opt_type==:min
      # this_best = sorted.last.take(2)
      # NArray[*sorted.map(&:last)]

      sort_idxs = fits.sort_index
      sort_idxs = sort_idxs.reverse if opt_type == :min
      this_best = [fits[sort_idxs[-1]], inds[sort_idxs[-1], true]]

      opt_cmp_fn = opt_type==:min ? :< : :>
      @best = this_best if this_best.first.send(opt_cmp_fn, best.first)

      samples[sort_idxs,true]
    end

    # @!method interface_methods
    # Declaring interface methods - implement these in child class!
    [:train, :initialize_distribution, :convergence].each do |mname|
      define_method mname do
        raise NotImplementedError, "Implement in child class!"
      end
    end
  end
end
