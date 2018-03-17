
module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Fixed Variance Natural Evolution Strategies
  class FNES < RNES

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      # g_sigma = utils.dot(picks.row_norms**2 - ndims).first # back to scalar
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
      # @variance *= Math.exp(g_sigma * lrate / 2)
      # @sigma = id * variance
    end
  end
end
