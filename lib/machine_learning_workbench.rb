
gpu = false             # prepare for switching to GPUs
if gpu
  require 'cumo/narray'
  require 'cumo/linalg'
  Xumo = Cumo
else
  require 'numo/narray'
  require 'numo/linalg'  # depends on openblas: `sudo apt install libopenblas-base`
  Xumo = Numo
end
NArray = Xumo::DFloat   # set a single data type across the WB for now
NMath = Xumo::NMath     # shorthand for extended math module

module MachineLearningWorkbench
  module Compressor
  end
  module NeuralNetwork
  end
  module Optimizer
  end
  module Tools
  end
end

require_relative 'machine_learning_workbench/monkey'
require_relative 'machine_learning_workbench/tools'
require_relative 'machine_learning_workbench/compressor'
require_relative 'machine_learning_workbench/neural_network'
require_relative 'machine_learning_workbench/optimizer'
