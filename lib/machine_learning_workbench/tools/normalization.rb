# frozen_string_literal: true

module MachineLearningWorkbench::Tools
  module Normalization
    def self.feature_scaling narr, from: nil, to: [0,1]
      from ||= narr.minmax
      old_min, old_max = from
      new_min, new_max = to
      ( (narr-old_min)*(new_max-new_min)/(old_max-old_min) ) + new_min
    rescue ZeroDivisionError
      # require 'pry'; binding.pry
      raise ArgumentError, "If you get here, chances are there's a bug in `from` or `to`"
    end

    # @param per_column [bool] wheather to compute stats per-column or matrix-wise
    def self.z_score narr, per_column: true
      raise NotImplementedError unless per_column
      raise "this would be a good time to test this implementation"
      means = narr.mean
      stddevs = narr.std
      # address edge case of zero variance
      stddevs.map! { |v| v.zero? ? 1 : v }
      mean_mat = means.repeat narr.rows, 0
      stddev_mat = stddevs.repeat narr.rows, 0
      (narr - mean_mat) / stddev_mat
    end
  end
end
