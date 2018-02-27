
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

Gem::Specification.new do |spec|
  spec.name          = "machine_learning_workbench"
  spec.version       = `git describe`
  spec.authors       = ["Giuseppe Cuccu"]
  spec.email         = ["giuseppe.cuccu@gmail.com"]

  spec.summary       = %q{Workbench for practical machine learning in Ruby.}
  spec.description   = %q{This workbench holds a collection of machine learning methods in Ruby. Rather than specializing on a single task or method, this gem aims at providing an encompassing framework for any machine learning application.}
  spec.homepage      = "https://github.com/giuse/machine_learning_workbench"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rspec", "~> 3.0"
  spec.add_development_dependency "pry", "~> 0.10"
  spec.add_development_dependency "pry-nav", "~> 0.2"
  spec.add_development_dependency "pry-rescue", "~> 1.4"
  spec.add_development_dependency "pry-stack_explorer", "~> 0.4"
end
