module MachineLearningWorkbench::Tools
  module Imaging
    Forkable = MachineLearningWorkbench::Tools::Execution
    Norm = MachineLearningWorkbench::Tools::Normalization

    # Create RMagick::Image from NMatrix data
    def self.nmat_to_img nmat, shape: nil
      shape ||= nmat.shape
      shape = [1, shape] if shape.kind_of?(Integer) || shape.size == 1
      # `Image::constitute` requires Float pixels to be in [0,1]
      pixels = Norm.feature_scaling nmat.round(4), to: [0,1]
      Magick::Image.constitute *shape, "I", pixels.to_flat_a
    end

    # Create PNG file from NMatrix data
    def self.nmat_to_png nmat, fname, shape: nil
      nmat_to_img(nmat, shape: shape).write fname
    end

    # Show a NMatrix as image in a RMagick window
    # @param disp_size the size of the image to display
    # @param shape the true shape of the image (NMatrix could be flattened)
    # @param in_fork [bool] whether to execute the display in fork (and continue running)
    def self.display nmat, disp_size: [300, 300], shape: nil, in_fork: true
      img = nmat_to_img(nmat, shape: shape).resize(*disp_size)
      if in_fork
        MachineLearningWorkbench::Tools::Execution.in_fork { img.display }
      else
        img.display
      end
    end

    # Create NMatrix from png by filename.
    # @param fname the file name
    # @param scale optional rescaling of the image
    # @param flat [bool] whether to return a flat array
    # @param dtype dtype for the NMatrix, leave `nil` for automatic detection
    def self.nmat_from_png fname, scale: nil, flat: false, dtype: nil
      img = Magick::ImageList.new(fname).first
      img.scale!(scale) if scale
      shape = [img.columns, img.rows]
      pixels = img.export_pixels(0, 0, *shape, 'I') # 'I' for intensity
      raise "Sanity check" unless shape.reduce(:*)==pixels.size
      return pixels.to_nm(nil, dtype) if flat
      NMatrix.new shape, pixels, dtype: dtype
    end
  end
end
