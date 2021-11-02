# [Machine Learning Workbench](https://github.com/giuse/machine_learning_workbench)

[![Gem Version](https://badge.fury.io/rb/machine_learning_workbench.svg)](https://badge.fury.io/rb/machine_learning_workbench)
[![Build Status](https://travis-ci.org/giuse/machine_learning_workbench.svg?branch=master)](https://travis-ci.org/giuse/machine_learning_workbench)
[![Code Climate](https://codeclimate.com/github/giuse/machine_learning_workbench/badges/gpa.svg)](https://codeclimate.com/github/giuse/machine_learning_workbench)

This workbench holds a collection of machine learning methods in Ruby. Rather than specializing on a single task or method, this gem aims at providing an encompassing framework for any machine learning application.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'machine_learning_workbench'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install machine_learning_workbench

## Usage

TLDR: Check out [the `examples` directory](examples), e.g. [this script](examples/neuroevolution.rb).

This library is thought as a practical workbench: there is plenty of tools hanging, each has multiple uses and applications, and as such it is built as atomic and flexible as possible. Folders [in the lib structure](lib/machine_learning_workbench) categorize them.

The [systems directory](lib/machine_learning_workbench/systems) holds a few examples of how to bring them together in higher abstractions, i.e. as _compound tools_.
For example, a [neuroevolution setup](lib/machine_learning_workbench/systems/neuroevolution.rb) brings together evolutionary computation and neural networks.

For an example of how to build it from scratch, check this [neuroevolution script](examples/neuroevolution.rb). To run it, use `bundle exec ruby examples/neuroevolution.rb`


## Development

After cloning the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/giuse/machine_learning_workbench.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## References

Please feel free to contribute to this list (see `Contributing` above).

- **NES** stands for Natural Evolution Strategies. Check its [Wikipedia page](https://en.wikipedia.org/wiki/Natural_evolution_strategy) for more info.
- **CMA-ES** stands for Covariance Matrix Adaptation Evolution Strategy. Check its [Wikipedia page](https://en.wikipedia.org/wiki/CMA-ES) for more info.
- **UL-ELR** stands for Unsupervised Learning plus Evolutionary Reinforcement Learning, from the paper _"Intrinsically Motivated Neuroevolution for Vision-Based Reinforcement Learning" (ICDL2011)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- **BD-NES** stands for Block Diagonal Natural Evolution Strategy, from the homonymous paper _"Block Diagonal Natural Evolution Strategies" (PPSN2012)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- **RNES** stands for Radial Natural Evolution Strategy, from the paper _"Novelty-Based Restarts for Evolution Strategies" (CEC2011)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- **DLR-VQ** stands for Decaying Learning Rate Vector Quantization, from the algorithm originally named _*Online VQ*_ in the paper _"Intrinsically Motivated Neuroevolution for Vision-Based Reinforcement Learning" (ICDL2011)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
