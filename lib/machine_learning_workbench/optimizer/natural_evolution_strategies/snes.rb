# frozen_string_literal: true

module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Separable Natural Evolution Strategies
  class SNES < Base

    attr_reader :variances

    def initialize_distribution mu_init: 0, sigma_init: 1
      @mu = case mu_init
        when Array
          raise ArgumentError unless mu_init.size == ndims
          NArray[mu_init]
        when Numeric
          NArray.new([1,ndims]).fill mu_init
        else
          raise ArgumentError, "Something is wrong with mu_init: #{mu_init}"
      end
      @variances = case sigma_init
      when Array
        raise ArgumentError unless sigma_init.size == ndims
        NArray[*sigma_init]
      when Numeric
        NArray.new([ndims]).fill(sigma_init)
      else
        raise ArgumentError, "Something is wrong with sigma_init: #{sigma_init}" \
          "(did you remember to copy the other cases from XNES?)"
      end
      @sigma = @variances.diag
    end

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      g_sigma = utils.dot(picks**2 - 1)
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
      @variances *= (g_sigma * lrate / 2).exponential.flatten
      @sigma = @variances.diag
    end

    # Estimate algorithm convergence as total variance
    def convergence
      variances.sum
    end

    def save
      [mu.to_a, variances.to_a]
    end

    def load data
      raise ArgumentError unless data.size == 2
      @mu, @variances = data.map &:to_na
      @sigma = variances.diag
    end
  end
end
