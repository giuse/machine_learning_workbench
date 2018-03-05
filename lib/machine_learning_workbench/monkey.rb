
# Monkey patches

module Monkey
  module Dimensionable
    def dims ret: []
      ret << size
      if first.kind_of? Array
        # hypothesize all elements having same size and save some checks
        first.dims ret: ret
      else
        ret
      end
    end
  end

  module Buildable
    def new *args
      super.tap do |m|
        if block_given?
          m.each_stored_with_indices do |_,*idxs|
            m[*idxs] = yield *idxs
          end
        end
      end
    end
  end

end

Array.include Monkey::Dimensionable
NMatrix.extend Monkey::Buildable
