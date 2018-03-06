VectorQuantization = MachineLearningWorkbench::Compressor::VectorQuantization
Img = MachineLearningWorkbench::Tools::Imaging
Norm = MachineLearningWorkbench::Tools::Normalization

RSpec.describe VectorQuantization, :SKIP do
  it "works" do
    require 'rmagick'

    ncentrs = 4
    image_files = Dir[ENV['HOME']+'/jaffe/KA.HA*.png']
    raise "Download the JAFFE dataset in your home dir" if image_files&.empty?
    # ... and convert the `.tiff` in `.png`: `mogrify -format png jaffe/*.tiff`
    centr_range = [-1, 1]
    orig_shape = [256, 256]
    img_range = [0, 2**16-1]

    puts "Loading images"
    images = image_files.map do |fname|
      nmat = Img.nmat_from_png fname, flat: true, dtype: :float64
      ret = Norm.feature_scaling nmat, from: img_range, to: centr_range
    end

    puts "Initializing VQ"
    vq = VectorQuantization.new ncentrs: ncentrs,
      dims: images.first.shape, lrate: 0.3,
      dtype: images.first.dtype, vrange: centr_range

    puts "Training"
    vq.train images, debug: true

    puts "Done!"
    begin
      vq.centrs.map { |c| Img.display c, shape: orig_shape }
      require 'pry'; binding.pry
      # Img.display vq.centrs.first, shape: orig_shape #, in_fork: false
    ensure
      MachineLearningWorkbench::Tools::Execution.kill_forks
    end
  end
end
