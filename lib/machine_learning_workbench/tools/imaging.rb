# frozen_string_literal: true

module MachineLearningWorkbench::Tools
  module Imaging
    Forkable = MachineLearningWorkbench::Tools::Execution
    Norm = MachineLearningWorkbench::Tools::Normalization

    # Create RMagick::Image from numeric matrix data
    # @param narr [NArray] numeric matrix to display
    # @param shape [Array<Integer>] optional reshaping
    def self.narr_to_img narr, shape: nil
      require 'rmagick'
      shape ||= narr.shape
      shape = [1, shape] if shape.kind_of?(Integer) || shape.size == 1
      # `Image::constitute` requires Float pixels to be in [0,1]
      pixels = Norm.feature_scaling narr.cast_to(NArray), to: [0,1]
      Magick::Image.constitute *shape, "I", pixels.to_a.flatten
    end

    # Create PNG file from numeric matrix data
    # @param narr [NArray] numeric matrix to display
    # @param fname [String] path to save PNG
    # @param shape [Array<Integer>] optional reshaping before saving
    def self.narr_to_png narr, fname, shape: nil
      narr_to_img(narr, shape: shape).write fname
    end

    # Show a numeric matrix as image in a RMagick window
    # @param narr [NArray] numeric matrix to display
    # @param disp_size [Array] the size of the image to display
    # @param shape [Array] the true shape of the image (numeric matrix could be flattened)
    # @param in_fork [bool] whether to execute the display in fork (and continue running)
    def self.display narr, disp_size: nil, shape: nil, in_fork: true
      require 'rmagick'
      img = narr_to_img narr, shape: shape
      img.resize!(*disp_size, Magick::TriangleFilter,0.51) if disp_size
      if in_fork
        MachineLearningWorkbench::Tools::Execution.in_fork { img.display }
      else
        img.display
      end
    end

    # Create numeric matrix from png by filename.
    # @param fname the file name
    # @param scale optional rescaling of the image
    # @param flat [bool] whether to return a flat array
    # @param dtype dtype for the numeric matrix, leave `nil` for automatic detection
    def self.narr_from_png fname, scale: nil, flat: false
      require 'rmagick'
      img = Magick::ImageList.new(fname).first
      img.scale!(scale) if scale
      shape = [img.columns, img.rows]
      pixels = img.export_pixels(0, 0, *shape, 'I') # 'I' for intensity
      raise "Sanity check" unless shape.reduce(:*)==pixels.size
      return pixels.to_na if flat
      pixels.to_na.to_dimensions shape
    end
  end
end
