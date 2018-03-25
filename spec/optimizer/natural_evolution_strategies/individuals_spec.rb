
RSpec.describe MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies do
  NES = MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies

  describe :inds do

    context "when sorted by fitness" do
      fit = lambda { |ind| ind.sum }

      context "with artificial inds" do
        inds = [[7,8,9], [1,2,3], [4,5,6]]
        a,b,c = inds
        max_sort = [b,c,a]
        min_sort = max_sort.reverse

        class TestNES < NES::Base
          def initialize_distribution mu_init: nil, sigma_init: nil
            @eye = NArray.eye(@ndims)
            @mu = NArray.zeros([1,@ndims])
            @sigma = @eye.copy
            @popsize = 3 # must match with `inds` declared above
          end
        end

        nes = TestNES.new(inds.first.size, fit, :min)
        nes.instance_eval("def standard_normal_samples; NArray[*#{inds}] end")

        it "minimization" do
          expect(nes.sorted_inds.to_a).to eq(min_sort)
          expect(nes.sorted_inds.to_a).not_to eq(max_sort)
          expect(nes.sorted_inds.to_a).not_to eq(inds)
        end

        it "maximization" do
          nes.instance_eval("@opt_type = :max")
          expect(nes.sorted_inds.to_a).to eq(max_sort)
          expect(nes.sorted_inds.to_a).not_to eq(min_sort)
          expect(nes.sorted_inds.to_a).not_to eq(inds)
        end
      end

      context "with generated inds" do
        ndims = 5
        nes = TestNES.new(ndims, fit, :min)
        # fetch individuals through nes sampling
        inds = nes.standard_normal_samples.to_a
        fits = inds.map &fit
        max_idx = fits.each_with_index.sort.map &:last
        max_sort = inds.values_at *max_idx
        min_sort = max_sort.reverse
        # fix the sampling to last sample
        nes.instance_eval("def standard_normal_samples; NArray[*#{inds}] end")
        # fix the sigma not to alter the ind
        nes.instance_eval("@sigma = @eye.copy")

        it "minimization" do
          expect(nes.sorted_inds.to_a).to eq(min_sort)
          expect(nes.sorted_inds.to_a).not_to eq(max_sort)
          expect(nes.sorted_inds.to_a == inds && inds != min_sort).to be_falsey
        end

        it "maximization" do
          nes.instance_eval("@opt_type = :max")
          expect(nes.sorted_inds.to_a).to eq(max_sort)
          expect(nes.sorted_inds.to_a).not_to eq(min_sort)
          expect(nes.sorted_inds.to_a == inds && inds != max_sort).to be_falsey
        end
      end
    end

  end
end
