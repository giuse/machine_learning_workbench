
module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Exponential Natural Evolution Strategies
  class XNES < Base
    attr_reader :log_sigma

    def initialize_distribution mu_init: 0, sigma_init: 1
      @mu = NMatrix.new([1, ndims], mu_init, dtype: dtype)
      sigma_init = [sigma_init]*ndims unless sigma_init.kind_of? Enumerable
      @sigma = NMatrix.diag(sigma_init, dtype: dtype)
      # Works with the log of sigma to avoid continuous decompositions (thanks Sun Yi)
      log_sigma_init = sigma_init.map &Math.method(:log)
      @log_sigma = NMatrix.diag(log_sigma_init, dtype: dtype)
    end

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      g_log_sigma = popsize.times.inject(NMatrix.zeros_like sigma) do |sum, i|
        u = utils[i]
        ind = picks.row(i)
        ind_sq = ind.outer_flat(ind, &:*)
        sum + (ind_sq - id) * u
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
      [mu.to_consistent_a, log_sigma.to_consistent_a]
    end

    def load data
      raise ArgumentError unless data.size == 2
      mu_ary, log_sigma_ary = data
      @mu = NMatrix[*mu_ary, dtype: dtype]
      @log_sigma = NMatrix[*log_sigma_ary, dtype: dtype]
      @sigma = log_sigma.exponential
    end
  end
end
