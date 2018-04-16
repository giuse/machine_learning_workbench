# frozen_string_literal: true

module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Radial Natural Evolution Strategies
  class RNES < Base
    attr_reader :variance

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
      @variance = sigma_init
      @sigma = case sigma_init
      when Array
        raise ArgumentError "RNES uses single global variance"
      when Numeric
        NArray.new([ndims]).fill(variance).diag
      else
        raise ArgumentError, "Something is wrong with sigma_init: #{sigma_init}"
      end
    end

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      # g_sigma = utils.dot(picks.row_norms**2 - ndims).first # back to scalar
      row_norms = NLinalg.norm picks, 2, axis:1
      g_sigma = utils.dot(row_norms**2 - ndims)[0] # back to scalar
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
      @variance *= Math.exp(g_sigma * lrate / 2)
      @sigma = NArray.new([ndims]).fill(variance).diag
    end

    # Estimate algorithm convergence based on variance
    def convergence
      variance
    end

    def save
      [mu.to_a, variance]
    end

    def load data
      raise ArgumentError unless data.size == 2
      mu_ary, @variance = data
      @mu = mu_ary.to_na
      @sigma = eye * variance
    end
  end
end
