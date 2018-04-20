# frozen_string_literal: true

require "bundler/setup"
require "machine_learning_workbench"
require_relative 'helpers/uses_temporary_folders'

STDOUT.sync = true

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"
  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # These two settings work together to allow you to limit a spec run
  # to individual examples or groups you care about by tagging them with
  # `:focus` metadata. When nothing is tagged with `:focus`, all examples
  # get run.
  config.filter_run :FOCUS
  config.filter_run_excluding :SKIP
  config.run_all_when_everything_filtered = true
end
