# frozen_string_literal: true

module MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  # Fixed Variance Natural Evolution Strategies
  class FNES < RNES

    def train picks: sorted_inds
      g_mu = utils.dot(picks)
      @mu += sigma.dot(g_mu.transpose).transpose * lrate
    end
  end
end
