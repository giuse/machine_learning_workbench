# frozen_string_literal: true

# Run as: `bundle exec ruby examples/image_compression.rb`

require 'rmagick'
require 'machine_learning_workbench'
VectorQuantization = MachineLearningWorkbench::Compressor::VectorQuantization
Img = MachineLearningWorkbench::Tools::Imaging
Norm = MachineLearningWorkbench::Tools::Normalization

ncentrs = 1
# image_files = Dir[ENV['HOME']+'/jaffe/KA.HA*.png']
image_files = Dir[ENV['HOME']+'/jaffe/*.png']
raise "Download the JAFFE dataset in your home dir" if image_files&.empty?
# ... and convert the `.tiff` in `.png`: `mogrify -format png jaffe/*.tiff`
centr_range = [-1, 1]
orig_shape = [256, 256]
img_range = [0, 2**16-1]

puts "Loading images"
images = image_files.map do |fname|
  ary = Img.narr_from_png fname, flat: true
  ret = Norm.feature_scaling ary, from: img_range, to: centr_range
end

puts "Initializing VQ"
vq = VectorQuantization.new ncentrs: ncentrs,
  dims: images.first.shape, lrate: 0.3, vrange: centr_range

puts "Training"
vq.train images, debug: true

puts "Done!"
begin
  vq.centrs.map { |c| Img.display c, shape: orig_shape }
  require 'pry'; binding.pry
ensure
  MachineLearningWorkbench::Tools::Execution.kill_forks
end
