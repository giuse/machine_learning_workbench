
gpu = false             # prepare for switching to GPUs
if gpu
  require 'cumo/narray'
  Xumo = Cumo
  require 'cumo/linalg'
else
  require 'numo/narray'
  Xumo = Numo
  # gem `numo-linalg` depends on openblas and lapacke:
  # `sudo apt install libopenblas-base liblapacke`
  require 'numo/linalg'
end

# Shorthands
NArray = Xumo::DFloat   # set a single data type across the WB for now
NMath = Xumo::NMath     # shorthand for extended math module
NLinalg = Xumo::Linalg  # shorthand for linear algebra module

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
WB = MachineLearningWorkbench # import MachineLearningWorkbench as WB ;)

require_relative 'machine_learning_workbench/monkey'
require_relative 'machine_learning_workbench/tools'
require_relative 'machine_learning_workbench/compressor'
require_relative 'machine_learning_workbench/neural_network'
require_relative 'machine_learning_workbench/optimizer'
