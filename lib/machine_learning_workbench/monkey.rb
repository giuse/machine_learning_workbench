
# Monkey patches

module MachineLearningWorkbench::Monkey
  module Dimensionable
    def dims ret: []
      ret << size
      if first.kind_of? Array
        # hypothesize all elements having same size and save some checks
        first.dims ret: ret
      else
        ret
      end
    end
  end

  module Buildable
    def new *args
      super.tap do |m|
        if block_given?
          m.each_stored_with_indices do |_,*idxs|
            m[*idxs] = yield *idxs
          end
        end
      end
    end
  end

  module AdvancelyOperationable # how am I supposed to name these things??

    # Outer matrix relationship generalization.
    # Make a matrix the same shape as `self`; each element is a matrix,
    # with the same shape as `other`, resulting from the interaction of
    # the corresponding element in `self` and all the elements in `other`.
    # @param other [NMatrix] other matrix
    # @note This implementation works only for 2D matrices (same as most
    #   other methods here). It's a quick hack, a proof of concept barely
    #   sufficient for my urgent needs.
    # @note Output size is fixed! Since NMatrix does not graciously yield to
    #   being composed of other NMatrices (by adapting the shape of the root
    #   matrix), the block cannot return matrices in there.
    # @return [NMatrix]
    def outer other
      # NOTE: Map of map in NMatrix does not work as expected!
      # self.map { |v1| other.map { |v2| yield(v1,v2) } }
      # NOTE: this doesn't cut it either... can't capture the structure
      # NMatrix[ *self.collect { |v1| other.collect { |v2| yield(v1,v2) } } ]
      raise ArgumentError unless block_given?
      NMatrix.new(self.shape+other.shape).tap do |m|
        each_stored_with_indices do |v1,r1,c1|
          other.each_stored_with_indices do |v2,r2,c2|
            m[r1,c1,r2,c2] = yield(v1,v2)
          end
        end
      end
    end

    # Flat-output generalized outer relationship. Same as `#outer`, but the
    # result is a 2-dim matrix of the interactions between all the elements
    # in `self` (as rows) and all the elements in `other` (as columns)
    # @param other [NMatrix] other matrix
    # @return [NMatrix]
    def outer_flat other
      raise ArgumentError unless block_given?
      data = collect { |v1| other.collect { |v2| yield(v1, v2) } }
      self.class[*data, dtype: dtype]
    end

    # Matrix exponential: `e^self` (not to be confused with `self^n`!)
    # @return [NMatrix]
    def exponential
      # special case: one-dimensional matrix: just exponentiate the values
      if (dim == 1) || (dim == 2 && shape.include?(1))
        return NMatrix.new shape, collect(&Math.method(:exp)), dtype: dtype
      end

      # Eigenvalue decomposition method from scipy/linalg/matfuncs.py#expm2

      # TODO: find out why can't I get away without double transpose!
      e_values, e_vectors = eigen_symm

      e_vals_exp_dmat = NMatrix.diagonal e_values.collect(&Math.method(:exp))
      # ASSUMING WE'RE ONLY USING THIS TO EXPONENTIATE LOG_SIGMA IN XNES
      # Theoretically we need the right eigenvectors, which for a symmetric
      # matrix should be just transposes of the eigenvectors.
      # But we have a positive definite matrix, so the final composition
      # below holds without transposing
      # BUT, strangely, I can't seem to get eigen_symm to green the tests
      # ...with or without transpose
      # e_vectors = e_vectors.transpose
      e_vectors.dot(e_vals_exp_dmat).dot(e_vectors.invert)#.transpose
    end

    # Calculate matrix eigenvalues and eigenvectors using LAPACK
    # @param which [:both, :left, :right] which eigenvectors do you want?
    # @return [Array<NMatrix, NMatrix[, NMatrix]>]
    #   eigenvalues (as column vector), left eigenvectors, right eigenvectors.
    #   A value different than `:both` for param `which` reduces the return size.
    # @note requires LAPACK
    # @note WARNING! a param `which` different than :both alters the returns
    # @note WARNING! machine-precision-error imaginary part Complex
    # often returned! For symmetric matrices use #eigen_symm_right below
    def eigen which=:both
      raise ArgumentError unless [:both, :left, :right].include? which
      NMatrix::LAPACK.geev(self, which)
    end

    # Eigenvalues and right eigenvectors for symmetric matrices using LAPACK
    # @note code taken from gem `nmatrix-atlas` NMatrix::LAPACK#geev
    # @note FOR SYMMETRIC MATRICES ONLY!!
    # @note WARNING: will return real matrices, imaginary parts are discarded!
    # @note WARNING: only left eigenvectors will be returned!
    # @todo could it be possible to save some of the transpositions?
    # @return [Array<NMatrix, NMatrix>] eigenvalues and (left) eigenvectors
    def eigen_symm
      # TODO: check for symmetry if not too slow
      raise TypeError, "Only real-valued matrices" if complex_dtype?
      raise StorageTypeError, "Only dense matrices (because LAPACK)" unless dense?
      raise ShapeError, "Only square matrices" unless dim == 2 && shape[0] == shape[1]

      n = shape[0]

      # Outputs
      e_values = NMatrix.new([n, 1], dtype: dtype)
      e_values_img = NMatrix.new([n, 1], dtype: dtype) # to satisfy C alloc
      e_vectors = clone_structure

      NMatrix::LAPACK::lapack_geev(
        false,        # compute left eigenvectors of A?
        :t,           # compute right eigenvectors of A? (left eigenvectors of A**T)
        n,            # order of the matrix
        transpose,    # input matrix => needs to be column-wise  # self,
        n,            # leading dimension of matrix
        e_values,     # real part of computed eigenvalues
        e_values_img, # imaginary part of computed eigenvalues (will be discarded)
        nil,          # left eigenvectors, if applicable
        n,            # leading dimension of left_output
        e_vectors,    # right eigenvectors, if applicable
        n,            # leading dimension of right_output
        2*n           # no clue what's this
      )

      raise "Uhm why complex eigenvalues?" if e_values_img.any? {|v| v>1e-10}
      return [e_values, e_vectors.transpose]
    end


    # The NMatrix documentation refers to a function `#nrm2` (aliased to `#norm2`)
    # to compute the norm of a matrix. Fun fact: that is the implementation for vectors,
    # and calling it on a matrix returns NotImplementedError :) you have to toggle the
    # source to understand why:
    # http://sciruby.com/nmatrix/docs/NMatrix.html#method-i-norm2 .
    # A search for the actual source on GitHub reveals a (I guess new?) method
    # `#matrix_norm`, with a decent choice of norms to choose from. Unfortunately, as the
    # name says, it is stuck to compute full-matrix norms.
    # So I resigned to dance to `Array`s and back, and implemented it with `#each_rank`.
    # Unexplicably, I get a list of constant values as the return value; same with
    # `#each_row`.
    # What can I say, we're back to referencing rows by index. I am just wasting too much
    # time figuring out these details to write a generalized version with an optional
    # `dimension` to go along.
    # @return [NMatrix] the vector norm along the rows
    def row_norms
      norms = rows.times.map { |i| row(i).norm2 }
      NMatrix.new [rows, 1], norms, dtype: dtype
    end

    # `NMatrix#to_a` has inconsistent behavior: single-row matrices are
    # converted to one-dimensional Arrays rather than a 2D Array with
    # only one row. Patching `#to_a` directly is not feasible as the
    # constructor seems to depend on it, and I have little interest in
    # investigating further.
    # @return [Array<Array>] a consistent array representation, such that
    #   `nmat.to_consistent_a.to_nm == nmat` holds for single-row matrices
    def to_consistent_a
      dim == 2 && shape[0] == 1 ? [to_a] : to_a
    end
    alias :to_ca :to_consistent_a
  end

  module NumericallyApproximatable
    # Verifies if `self` and `other` are withing `epsilon` of each other.
    # @param other [Numeric]
    # @param epsilon [Numeric]
    # @return [Boolean]
    def approximates? other, epsilon=1e-5
      # Used for testing and NMatrix#approximates?, should I move to spec_helper?
      (self - other).abs < epsilon
    end
  end

  module MatrixApproximatable
    # Verifies if all values at corresponding indices approximate each other.
    # @param other [NMatrix]
    # @param epsilon [Float]
    def approximates? other, epsilon=1e-5
      return false unless self.shape == other.shape
      # two ways to go here:
      # - epsilon is aggregated: total cumulative accepted error
      #   => `(self - other).reduce(:+) < epsilon`
      # - epsilon is local: per element accepted error
      #   => `v.approximates? other[*idxs], epsilon`
      # Given the use I make (near-equality), I choose the first interpretation
      # Note the second is sensitive to opposite signs balancing up
      self.each_stored_with_indices.all? do |v,*idxs|
        v.approximates? other[*idxs], epsilon
      end
    end
  end

  module CPtrDumpable
    def marshall_dump
      [shape, dtype, data_pointer]
    end

    def marshall_load
      raise NotImplementedError, "There's no setter for the data pointer!"
    end
  end
end

Array.include MachineLearningWorkbench::Monkey::Dimensionable
NMatrix.extend MachineLearningWorkbench::Monkey::Buildable
require 'nmatrix/lapack_plugin' # loads whichever is installed between atlas and lapacke
NMatrix.include MachineLearningWorkbench::Monkey::AdvancelyOperationable
Numeric.include MachineLearningWorkbench::Monkey::NumericallyApproximatable
NMatrix.include MachineLearningWorkbench::Monkey::MatrixApproximatable
NMatrix.include MachineLearningWorkbench::Monkey::CPtrDumpable
