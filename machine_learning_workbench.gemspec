
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

Gem::Specification.new do |spec|
  spec.name          = "machine_learning_workbench"
  spec.version       = `git describe`
  spec.author        = "Giuseppe Cuccu"
  spec.email         = "giuseppe.cuccu@gmail.com"

  spec.summary       = %q[Workbench for practical machine learning in Ruby.]
  spec.description   = %q[This workbench holds a collection of machine learning
    methods in Ruby. Rather than specializing on a single task or method, this
    gem aims at providing an encompassing framework for any machine learning
    application.].gsub('  ', '')
  spec.homepage      = "https://github.com/giuse/machine_learning_workbench"
  spec.license       = "MIT"
  spec.post_install_message = %Q[\
    Thanks for installing the machine learning workbench!
    It is still a work in progress, feel free to open an issue or drop me an email
    and start a discussion if you are using this gem. Cheers!
  ].gsub('  ', '')

  spec.files = `git ls-files -z`.split("\x0").reject { |f| f.start_with? "spec" }

  # spec.bindir        = "exe"
  # spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.required_ruby_version = '>= 2.4.0'

  # Install
  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"

  # Test
  spec.add_development_dependency "rspec", "~> 3.0"
  spec.add_development_dependency "rmagick"  # uhm would gladly drop this

  # Debug
  spec.add_development_dependency "pry", "~> 0.10"
  spec.add_development_dependency "pry-nav", "~> 0.2"
  spec.add_development_dependency "pry-rescue", "~> 1.4"
  spec.add_development_dependency "pry-stack_explorer", "~> 0.4"
  spec.add_development_dependency "pry-doc", "~> 0.12"

  # Run
  spec.requirements << "libopenblas-base"  # library for following dependency
  spec.add_dependency "numo-narray", "~> 0.9"
  spec.add_dependency "numo-linalg", "~> 0.1"
  spec.add_dependency "parallel", "~> 1.12"
end
