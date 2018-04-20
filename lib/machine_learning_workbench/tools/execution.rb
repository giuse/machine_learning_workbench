# frozen_string_literal: true

module MachineLearningWorkbench::Tools
  module Execution
    $fork_pids ||= []

    # Executes block in a (detached) fork, saving the `pid` for later termination.
    # @note add `ensure MachineLearningWorkbench::Tools.kill_forks` to the block
    #    where `in_fork` is called (see `#kill_forks`).
    def self.in_fork &block
      raise ArgumentError "Need block to be executed in fork" unless block
      pid = fork(&block)
      Process.detach pid
      $fork_pids << pid
    end

    # Kills processes spawned by `#in_fork`.
    # Call this in an `ensure` block after using `in_fork`.
    # => `ensure MachineLearningWorkbench::Tools.kill_forks`
    def self.kill_forks
      $fork_pids&.each { |pid| Process.kill('KILL', pid) rescue Errno::ESRCH }
      $fork_pids = []
    end
  end
end
