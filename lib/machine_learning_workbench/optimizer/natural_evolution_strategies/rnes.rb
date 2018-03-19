
module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Radial Natural Evolution Strategies
  class RNES < Base
    attr_reader :variance

    def initialize_distribution mu_init: 0, sigma_init: 1
      @mu = NMatrix.new([1, ndims], mu_init, dtype: dtype)
      raise ArgumentError unless sigma_init.kind_of? Numeric
      @variance = sigma_init
      @sigma = id * variance
    end

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      g_sigma = utils.dot(picks.row_norms**2 - ndims).first # back to scalar
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
      @variance *= Math.exp(g_sigma * lrate / 2)
      @sigma = id * variance
    end

    # Estimate algorithm convergence based on variance
    def convergence
      variance
    end

    def save
      [mu.to_consistent_a, variance]
    end

    def load data
      raise ArgumentError unless data.size == 2
      mu_ary, @variance = data
      @mu = NMatrix[*mu_ary, dtype: dtype]
      @sigma = id * variance
    end
  end
end
