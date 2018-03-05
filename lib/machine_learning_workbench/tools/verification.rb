module MachineLearningWorkbench::Tools
  module Verification
    def self.in_range! nmat, vrange
      vmin, vmax = vrange.to_a
      nmat.each_with_indices do |v, *idxs|
        nmat[*idxs] = vmin if v < vmin
        nmat[*idxs] = vmax if v > vmax
      end
    end
  end
end
