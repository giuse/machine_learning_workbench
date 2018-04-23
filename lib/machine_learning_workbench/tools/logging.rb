# frozen_string_literal: true

module MachineLearningWorkbench::Tools
  module Logging
    # Splits calls to standard streams to be both displayed on terminal and saved to file
    class LogSplitter < File
      def initialize dest
        fname = if File.directory?(dest)
          "#{dest}/#{Time.now.strftime "%y%m%d_%H%M"}.log"
        else dest
        end
        super fname, 'w'
      end

      def write *args
        STDOUT.write *args
        super
      end
    end

    def self.split_to dest, also_stderr: false
      $stdout = LogSplitter.new dest
      $stderr = $stdout if also_stderr
    end

    def self.restore_streams
      $stdout.close
      $stdout = STDOUT
      $stderr = STDERR
    end
  end
end
