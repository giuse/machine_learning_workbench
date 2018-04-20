# frozen_string_literal: true

module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Exponential Natural Evolution Strategies
  class XNES < Base
    attr_reader :log_sigma

    def initialize_distribution mu_init: 0, sigma_init: 1
      @mu = case mu_init
        when Array
          raise ArgumentError unless mu_init.size == ndims
          NArray[mu_init]
        when Numeric
          NArray.new([1,ndims]).fill mu_init
        when NArray
          raise ArgumentError unless mu_init.size == ndims
          mu_init.ndim < 2 ? mu_init.reshape(1, ndims) : mu_init
        else
          raise ArgumentError, "Something is wrong with mu_init: #{mu_init}"
      end
      @sigma = case sigma_init
        when Array
          raise ArgumentError unless sigma_init.size == ndims
          NArray[*sigma_init].diag
        when Numeric
          NArray.new([ndims]).fill(sigma_init).diag
        when NArray
          raise ArgumentError unless sigma_init.size == ndims**2
          sigma_init.ndim < 2 ? sigma_init.reshape(ndims, ndims) : sigma_init
        else
          raise ArgumentError, "Something is wrong with sigma_init: #{sigma_init}"
      end
      # Works with the log of sigma to avoid continuous decompositions (thanks Sun Yi)
      @log_sigma = NMath.log(sigma.diagonal).diag
    end

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      g_log_sigma = popsize.times.inject(NArray.zeros sigma.shape) do |sum, i|
        u = utils[i]
        ind = picks[i, true]
        ind_sq = ind.outer_flat(ind, &:*)
        sum + (ind_sq - eye) * u
      end
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
      @log_sigma += g_log_sigma * (lrate/2)
      @sigma = log_sigma.exponential
    end

    # Estimate algorithm convergence as total variance
    def convergence
      sigma.trace
    end

    def save
      [mu.to_a, log_sigma.to_a]
    end

    def load data
      raise ArgumentError unless data.size == 2
      @mu, @log_sigma = data.map &:to_na
      @sigma = log_sigma.exponential
    end
  end
end
