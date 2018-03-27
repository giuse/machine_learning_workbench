
RSpec.describe MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies do
  NES = MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies

  describe :inds do

    context "when sorted by fitness" do
      fit = lambda { |ind| ind.sum }

       class TestNES < NES::Base
        def initialize_distribution mu_init: nil, sigma_init: nil
          @eye = NArray.eye(@ndims)
          @mu = NArray.zeros([1,@ndims])
          @sigma = @eye.copy
          @popsize = 3 # must match with `inds` declared above
        end
      end

      context "with generated inds" do
        ndims = 5
        nes = TestNES.new(ndims, fit, :min)
        # fetch individuals through nes sampling

        it "minimization" do
          nes_sums = nes.sorted_inds.sum(1)
          expect(nes_sums).to eq(nes_sums.sort.reverse)
        end

        it "maximization" do
          nes.instance_eval("@opt_type = :max")
          nes_sums = nes.sorted_inds.sum(1)
          expect(nes_sums).to eq(nes_sums.sort)
        end
      end
    end

  end
end
