module MachineLearningWorkbench::Tools
  module Normalization
    def self.feature_scaling nmat, from: nil, to: [0,1]
      from ||= nmat.minmax
      old_min, old_max = from
      new_min, new_max = to
      (nmat-old_min)*(new_max-new_min)/(old_max-old_min)+new_min
    end

    # @param per_column [bool] wheather to compute stats per-column or matrix-wise
    def self.z_score nmat, per_column: true
      raise NotImplementedError unless per_column
      means = nmat.mean
      stddevs = nmat.std
      # address edge case of zero variance
      stddevs.map! { |v| v.zero? ? 1 : v }
      mean_mat = means.repeat nmat.rows, 0
      stddev_mat = stddevs.repeat nmat.rows, 0
      (nmat - mean_mat) / stddev_mat
    end
  end
end
