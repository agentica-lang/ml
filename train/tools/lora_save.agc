## Writes the PEFT adapter of `test/fixtures/lora_peft/` again with `Lora.save` into DIR, for the PEFT side of the
## round trip: `python -I test/lora_golden.py check DIR` then loads DIR with PEFT. Manual; from `train/`:
##
## ```sh
## agentica tools/lora_save.agc DIR
## ```

import Lora from "../lora"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  args = System.args()
  if args.size != 1
    puts("usage: lora_save.agc DIR")
    System.exit(2)
  end

  case written(args[0])
    ok(_) -> puts("wrote #{args[0]}")
    error(e) ->
      puts("lora_save: #{e.message}")
      System.exit(1)
  end
end

private

def written(dir: String) -> Result[Void]
  Lora.save(dir, try Lora.open("test/fixtures/lora_peft", Tensor.Backend.Reference))
end
