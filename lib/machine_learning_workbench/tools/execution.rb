module MachineLearningWorkbench::Tools
  module Execution
    $fork_pids ||= []

    # Execute block in a fork. Be sure to check also `#kill_forks`
    def self.in_fork &block
      raise ArgumentError "Need block to be executed in fork" unless block
      pid = fork(&block)
      Process.detach pid
      $fork_pids << pid
    end

    # Call this in an `ensure` block after using `in_fork`
    def self.kill_forks
      $fork_pids&.each { |pid| Process.kill 'KILL', pid }
    end
  end
end
