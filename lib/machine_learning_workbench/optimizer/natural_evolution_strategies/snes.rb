
module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Separable Natural Evolution Strategies
  class SNES < Base

    attr_reader :variances

    def initialize_distribution mu_init: 0, sigma_init: 1
      @mu = NMatrix.new([1, ndims], mu_init, dtype: dtype)
      sigma_init = [sigma_init]*ndims unless sigma_init.kind_of? Enumerable
      @variances = NMatrix.new([1,ndims], sigma_init, dtype: dtype)
      @sigma = NMatrix.diagonal(variances, dtype: dtype)
    end

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      g_sigma = utils.dot(picks**2 - 1)
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
      @variances *= (g_sigma * lrate / 2).exponential
      @sigma = NMatrix.diagonal(variances, dtype: dtype)
    end

    # Estimate algorithm convergence as total variance
    def convergence
      variances.reduce :+
    end

    def save
      [mu.to_consistent_a, variances.to_consistent_a]
    end

    def load data
      raise ArgumentError unless data.size == 2
      mu_ary, variances_ary = data
      @mu = NMatrix[*mu_ary, dtype: dtype]
      @variances = NMatrix[*variances_ary, dtype: dtype]
      @sigma = NMatrix.diagonal(variances, dtype: dtype)
    end
  end
end
