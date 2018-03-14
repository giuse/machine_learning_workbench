# evaluate in temporary (empty) folder
module UsesTemporaryFolders
  def self.included example_group
    example_group.extend self
  end

  def in_temporary_folder
    require 'pathname'
    attr_reader :orig_dir, :tmp_dir
    # ensure working in empty temporary folder
    before do
      @orig_dir = Pathname.pwd
      @tmp_dir = orig_dir + "in_temporary_folder"
      FileUtils.rm_rf tmp_dir
      FileUtils.mkdir_p tmp_dir
      Dir.chdir tmp_dir
    end
    # clean up
    after do
      Dir.chdir orig_dir
      FileUtils.rm_rf tmp_dir
    end
  end
end
