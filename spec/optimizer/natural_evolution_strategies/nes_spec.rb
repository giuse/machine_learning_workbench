
RSpec.describe MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies do
  NES = MachineLearningWorkbench::Optimizer::NaturalEvolutionStrategies
  ndims = 5          # XNES, SNES RNES
  ndims_lst = [3,2]  # BDNES
  obj_fns = {
    # MINIMIZATION: upper parabolic with minimum in [0]*ndims
    min: -> (ind) { ind.inject(0) { |mem,var| mem + var**2 } },
    # MAXIMIZATION: lower parabolic with maximum in [0]*ndims
    max: -> (ind) { ind.inject(0) { |mem,var| mem - var**2 } }
  }
  opt_types=obj_fns.keys
  one_opt_type = opt_types.first
  ntrains = 180

  describe NES::XNES do

    describe "#init" do
      it "initializes correctly" do
        opt_type = opt_types.sample # try either :)
        nes = NES::XNES.new ndims, obj_fns[opt_type], opt_type

        expect(opt_types).to include nes.opt_type
        expect(nes.obj_fn).to eq(obj_fns[nes.opt_type])
      end
    end

    describe "#train" do
      describe "full run" do
        opt_type = opt_types.sample # try either :)
        nes = NES::XNES.new ndims, obj_fns[opt_type], opt_type, rseed: 1
        context "within #{ntrains} iterations" do
          it "optimizes the negative squares function" do
            ntrains.times { nes.train }
            expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
            expect(nes.convergence.approximates? 0).to be_truthy
          end
        end
      end

      describe "with parallel fit" do
        opt_type = opt_types.sample # try either :)
        fit_par = -> (inds) { inds.map &obj_fns[opt_type] }
        nes = NES::XNES.new ndims, fit_par, opt_type, parallel_fit: true, rseed: 1
        context "within #{ntrains} iterations" do
          it "optimizes the negative squares function" do
            ntrains.times { nes.train }
            expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
            expect(nes.convergence.approximates? 0).to be_truthy
          end
        end
      end
    end

    describe "resuming" do
      it "#dump and #load" do
        a = NES::XNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 1
        3.times { a.train }
        a_dump = a.save
        b = NES::XNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 2
        b.load a_dump
        b_dump = b.save
        expect(a_dump).to eq(b_dump)
      end

      it "#load allows resuming" do
        nes = NES::XNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 1
        4.times { nes.train }
        run_4_straight = nes.save

        nes = NES::XNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 1
        2.times { nes.train }
        run_2_only = nes.save

        # If I resume with a new nes, it works, but results differ because
        # it changes the number of times the rand has been sampled
        nes_new = NES::XNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 1
        nes_new.load run_2_only
        2.times { nes_new.train }
        run_4_resumed_new = nes_new.save
        expect(run_4_straight).not_to eq(run_4_resumed_new)

        # If instead I use a nes with same rseed and same number of rand
        # calls, even though I trash the dist info, it yields the same result
        nes.load run_2_only
        2.times { nes.train }
        run_4_resumed = nes.save
        expect(run_4_straight).to eq(run_4_resumed)
      end
    end
  end

  describe NES::SNES do
    describe "full run" do
      opt_type = opt_types.sample # try either :)
      nes = NES::SNES.new ndims, obj_fns[opt_type], opt_type, rseed: 1
      context "within #{ntrains} iterations" do
        it "optimizes the negative squares function" do
          ntrains.times { nes.train }
          expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
          expect(nes.convergence.approximates? 0).to be_truthy
        end
      end
    end

    describe "with parallel fit" do
      opt_type = opt_types.sample # try either :)
      fit_par = -> (inds) { inds.map &obj_fns[opt_type] }
      nes = NES::SNES.new ndims, fit_par, opt_type, parallel_fit: true, rseed: 1
      context "within #{ntrains} iterations" do
        it "optimizes the negative squares function" do
          ntrains.times { nes.train }
          expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
          expect(nes.convergence.approximates? 0).to be_truthy
        end
      end
    end

    describe "resuming" do
      it "#dump and #load" do
        a = NES::SNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 1
        3.times { a.train }
        a_dump = a.save
        b = NES::SNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 2
        b.load a_dump
        b_dump = b.save
        expect(a_dump).to eq(b_dump)
      end
    end
  end

  describe NES::RNES do
    describe "full run" do
      opt_type = opt_types.sample # try either :)
      nes = NES::RNES.new ndims, obj_fns[opt_type], opt_type, rseed: 1
      context "within #{ntrains} iterations" do
        it "optimizes the negative squares function" do
          ntrains.times { nes.train }
          expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
          expect(nes.convergence.approximates? 0).to be_truthy
        end
      end
    end

    describe "with parallel fit" do
      opt_type = opt_types.sample # try either :)
      fit_par = -> (inds) { inds.map &obj_fns[opt_type] }
      nes = NES::RNES.new ndims, fit_par, opt_type, parallel_fit: true, rseed: 1
      context "within #{ntrains} iterations" do
        it "optimizes the negative squares function" do
          ntrains.times { nes.train }
          expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
          expect(nes.convergence.approximates? 0).to be_truthy
        end
      end
    end

    describe "resuming" do
      it "#dump and #load" do
        a = NES::RNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 1
        3.times { a.train }
        a_dump = a.save
        b = NES::RNES.new ndims, obj_fns[one_opt_type], one_opt_type, rseed: 2
        b.load a_dump
        b_dump = b.save
        expect(a_dump).to eq(b_dump)
      end
    end
  end

  describe NES::BDNES do
    describe "full run" do
      opt_type = opt_types.sample # try either :)
      nes = NES::BDNES.new [3,2], obj_fns[opt_type], opt_type, rseed: 1
      context "within #{ntrains} iterations" do
        it "optimizes the negative squares function" do
          ntrains.times { nes.train }
          expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
          expect(nes.convergence.approximates? 0).to be_truthy
        end
      end
    end

    describe "with parallel fit" do
      opt_type = opt_types.sample # try either :)
      fit_par = -> (inds) { inds.map &obj_fns[opt_type] }
      nes = NES::BDNES.new [3,2], fit_par, opt_type, parallel_fit: true, rseed: 1
      context "within #{ntrains} iterations" do
        it "optimizes the negative squares function" do
          ntrains.times { nes.train }
          expect(nes.mu.all? { |v| v.approximates? 0 }).to be_truthy
          expect(nes.convergence.approximates? 0).to be_truthy
        end
      end
    end

    describe "resuming" do
      it "#dump and #load" do
        a = NES::BDNES.new ndims_lst, obj_fns[one_opt_type], one_opt_type, rseed: 1
        3.times { a.train }
        a_dump = a.save
        b = NES::BDNES.new ndims_lst, obj_fns[one_opt_type], one_opt_type, rseed: 2
        b.load a_dump
        b_dump = b.save
        expect(a_dump).to eq(b_dump)
      end
    end
  end

end
