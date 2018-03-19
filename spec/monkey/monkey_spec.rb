RSpec.describe MachineLearningWorkbench::Monkey do

  describe Numeric do
    v = 1e-3
    describe "#approximates?" do
      it do
        expect(v.approximates? v+1e-4, 1e-3).to be_truthy
        expect(v.approximates? v+1e-2, 1e-3).to be_falsey
      end
    end
  end

  describe NMatrix do
    data = [[1,2,3],[4,5,6],[7,8,9]]
    nmat = NMatrix[*data, dtype: :float64]
    diag = [1,5,9]

    it "::new with a block" do
      shape = [data.size, data.first.size]
      built = NMatrix.new(shape) { |i,j| data[i][j]**2 }
      expect(built).to eq(nmat**2)
    end

    context "when looping on the diagonal", :SKIP do
      it "#each_diag" do
        expect(nmat.each_diag.to_a).to_eq(diag.collect {|n| NMatrix[[n]]})
      end

      it "#each_stored_diag" do
        expect(nmat.each_stored_diag.to_a).to eq(diag)
      end
    end

    context "when setting the diagonal", :SKIP do
      set_diag_diag = [10,50,90]
      set_diag_data = [[10,2,3],[4,50,6],[7,8,90]]
      set_diag_nmat = NMatrix[*set_diag_data]

      it "#set_diag" do
        setted = nmat.set_diag {|i| set_diag_diag[i]}
        expect(setted).to eq(set_diag_nmat)
        expect(nmat).not_to eq(setted)
        expect(nmat.object_id).not_to eq(setted.object_id)
      end

      it "#set_diag!" do
        tmp_mat = nmat.clone
        setted = tmp_mat.set_diag! {|i| set_diag_diag[i]}
        expect(setted).to eq(set_diag_nmat)
        expect(tmp_mat).to eq(setted)
      end

    end

    describe "#outer" do
      mini = NMatrix[[1,2],[3,4]]
      exp = NMatrix[[[[2, 3], [4, 5]],
                     [[3, 4], [5, 6]]],
                    [[[4, 5], [6, 7]],
                     [[5, 6], [7, 8]]]]
      it "computes the correct result" do
        res = mini.outer(mini) {|a,b| a+b}
        expect(res.shape).to eq(exp.shape)
        expect(res).to eq(exp)
      end
    end

    describe "#outer_flat" do
      mini = NMatrix[[1,2],[3,4]]
      exp_flat = NMatrix[[2, 3, 4, 5],
                         [3, 4, 5, 6],
                         [4, 5, 6, 7],
                         [5, 6, 7, 8]]
      it "computes the correct result" do
        res = mini.outer_flat(mini) {|a,b| a+b}
        expect(res.shape).to eq(exp_flat.shape)
        expect(res).to eq(exp_flat)
      end
    end

    describe "#eigen" do
      trg_eigenvalues = NMatrix[[16.11684, -1.11684, 0.0]].transpose
      trg_eigenvectors = NMatrix[[0.283349, 0.641675, 1.0],
                                 [-1.28335, -0.141675, 1.0],
                                 [1.0, -2.0, 1.0]].transpose
      # NMatrix (LAPACK) -- e_values, left_e_vecs, right_e_vecs
      eigenvalues, _, eigenvectors = nmat.eigen

      def eigencheck? orig, e_vals, e_vecs
        # INPUT: original matrix, eigenvalues accessible by index,
        #        NMatrix with corresponding eigenvectors in columns
        e_vecs.each_column.each_with_index.all? do |e_vec_t, i|
          left = orig.dot(e_vec_t)
          right = e_vec_t * e_vals[i]
          left.approximates? right
        end
      end

      it "solves the eigendecomposition" do
        expect(eigencheck?(nmat, trg_eigenvalues, trg_eigenvectors)).to be_truthy
        expect(eigenvalues.approximates? trg_eigenvalues).to be_truthy
        expect(eigencheck?(nmat, eigenvalues, eigenvectors)).to be_truthy
      end
    end

    describe "#exponential" do
      testmat = nmat/10.0 # let's avoid 1e6 values, shall we?
      exp = [[1.37316, 0.531485, 0.689809],
             [1.00926, 2.24815, 1.48704],
             [1.64536, 1.96481, 3.28426]]
      it "computes the correct result" do
        left = testmat.exponential
        right = NMatrix[*exp]
        expect(left.approximates? right).to be_truthy
      end
    end

    describe "row_norms" do
      trg_row_norms = [[3.7416573], [8.7749643], [13.928388]]
      it "computes the correct result" do
        expect(nmat.row_norms.approximates? NMatrix[*trg_row_norms]).to be_truthy
      end
    end

    describe "#approximates?" do
      it do
        expect(nmat.approximates? nmat+1e-4, 1e-3).to be_truthy
        expect(nmat.approximates? nmat+1e-2, 1e-3).not_to be_truthy
      end
    end

    describe "#sort_rows_by" do
      it "should be implemented! And used in NES#sorted_inds!"
    end

    describe "#hjoin", :SKIP do
      it "should work with smaller matrices" do
        a = NMatrix.new([1,3], [1,2,3])
        b = NMatrix.new([1,2], [4,5])
        expect(a.hjoin(b)).to eq(NMatrix.new([1,5], [1,2,3,4,5]))
      end
      it "should work with larger matrices" do
        a = NMatrix.new([1,3], [1,2,3])
        b = NMatrix.new([1,4], [4,5,6,7])
        expect(a.hjoin(b)).to eq(NMatrix.new([1,7], [1,2,3,4,5,6,7]))
      end
      # it "should be tested also with multirow matrices"
    end

    describe "#vjoin", :SKIP do
      it "should work with smaller matrices" do
        a = NMatrix.new([3,1], [1,2,3])
        b = NMatrix.new([2,1], [4,5])
        expect(a.vjoin(b)).to eq(NMatrix.new([5,1], [1,2,3,4,5]))
      end
      it "should work with larger matrices" do
        a = NMatrix.new([3,1], [1,2,3])
        b = NMatrix.new([4,1], [4,5,6,7])
        expect(a.vjoin(b)).to eq(NMatrix.new([7,1], [1,2,3,4,5,6,7]))
      end
      # it "should be tested also with multicolumn matrices!"
    end

    describe "#to_consistent_a" do
      it "should always return an array with the same shape as the matrix" do
        { [2,2] => [[1,2],[3,4]],        # square
          [2,3] => [[1,2,3],[4,5,6]],    # rectangular (h)
          [3,2] => [[1,2],[3,4],[5,6]],  # rectangular (v)
          [1,3] => [[1,2,3]],            # single row => THIS FAILS FOR `NMatrix#to_a`!
          [3,1] => [[1],[2],[3]],        # single column
          [3]   => [1,2,3]               # single-dimensional
        }.each do |shape, ary|
          expect(NMatrix.new(shape, ary.flatten).to_consistent_a).to eq ary
        end
      end
    end
  end
end

RSpec.describe "NMatrix inconsistencies, fixed in `Monkey`" do


  # IF ANY OF THESE TESTS FAIL, DROP THE MONKEY AND USE THESE METHODS!


  # method #to_a not consistent! => wrote true_to_a (fixing it breaks #new)
  describe "#to_a" do
    it "does not always return an array with the same shape as the matrix" do
      {
        # [2,2] => [[1,2],[3,4]],        # square
        # [2,3] => [[1,2,3],[4,5,6]],    # rectangular (h)
        # [3,2] => [[1,2],[3,4],[5,6]],  # rectangular (v)
        [1,3] => [[1,2,3]],            # single row => THIS FAILS FOR `NMatrix#to_a`!
        # [3,1] => [[1],[2],[3]],        # single column
        # [3]   => [1,2,3]               # single-dimensional
      }.each do |shape, ary|
        expect(NMatrix.new(shape, ary.flatten).to_a).not_to eq ary
      end
    end
  end

end
