
module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Natural Evolution Strategies base class
  class Base
    attr_reader :ndims, :mu, :sigma, :opt_type, :obj_fn, :parallel_fit, :id, :rng, :last_fits, :best, :rescale_popsize, :rescale_lrate, :dtype

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
    # @param dtype [NMatrix dtype] NMatrix dtype for all matrix computation
    def initialize ndims, obj_fn, opt_type, rseed: nil, mu_init: 0, sigma_init: 1, parallel_fit: false, rescale_popsize: 1, rescale_lrate: 1, dtype: :float64
      raise ArgumentError unless [:min, :max].include? opt_type
      raise ArgumentError unless obj_fn.respond_to? :call
      @ndims, @opt_type, @obj_fn, @parallel_fit = ndims, opt_type, obj_fn, parallel_fit
      @rescale_popsize, @rescale_lrate = rescale_popsize, rescale_lrate
      @id = NMatrix.identity(ndims, dtype: dtype)
      rseed ||= Random.new_seed
      # puts "NES rseed: #{s}"  # currently disabled
      @rng = Random.new rseed
      @best = [(opt_type==:max ? -1 : 1) * Float::INFINITY, nil]
      @last_fits = []
      @dtype = dtype
      initialize_distribution mu_init: mu_init, sigma_init: sigma_init
    end

    # Box-Muller transform: generates standard (unit) normal distribution samples
    # @return [Float] a single sample from a standard normal distribution
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
    # @return [NMatrix] scale-invariant utilities
    def cmaes_utilities
      # Algorithm equations are meant for fitness maximization
      # Match utilities with individuals sorted by INCREASING fitness
      log_range = (1..popsize).collect do |v|
        [0, Math.log(popsize.to_f/2 - 1) - Math.log(v)].max
      end
      total = log_range.reduce(:+)
      buf = 1.0/popsize
      vals = log_range.collect { |v| v / total - buf }.reverse
      NMatrix[vals, dtype: dtype]
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

    # Samples a standard normal distribution to construct a NMatrix of
    #   popsize multivariate samples of length ndims
    # @return [NMatrix] standard normal samples
    def standard_normal_samples
      NMatrix.new([popsize, ndims], dtype: dtype) { standard_normal_sample }
    end

    # Move standard normal samples to current distribution
    # @return [NMatrix] individuals
    def move_inds inds
      # TODO: can we reduce the transpositions?
      # sigma.dot(inds.transpose).map(&mu.method(:+)).transpose
      multi_mu = NMatrix[*inds.rows.times.collect {mu.to_a}, dtype: dtype].transpose
      (multi_mu + sigma.dot(inds.transpose)).transpose
      # sigma.dot(inds.transpose).transpose + inds.rows.times.collect {mu.to_a}.to_nm
    end

    # Sorted individuals
    # NOTE: Algorithm equations are meant for fitness maximization. Utilities need to be
    # matched with individuals sorted by INCREASING fitness. Then reverse order for minimization.
    # @return standard normal samples sorted by the respective individuals' fitnesses
    def sorted_inds
      samples = standard_normal_samples
      inds = move_inds(samples).to_a
      fits = parallel_fit ? obj_fn.call(inds) : inds.map(&obj_fn)
      # Quick cure for NaN fitnesses
      fits.map! { |x| x.nan? ? (opt_type==:max ? -1 : 1) * Float::INFINITY : x }
      @last_fits = fits # allows checking for stagnation
      sorted = [fits, inds, samples.to_a].transpose.sort_by(&:first)
      sorted.reverse! if opt_type==:min
      this_best = sorted.last.take(2)
      opt_cmp_fn = opt_type==:min ? :< : :>
      @best = this_best if this_best.first.send(opt_cmp_fn, best.first)
      NMatrix[*sorted.map(&:last), dtype: dtype]
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
