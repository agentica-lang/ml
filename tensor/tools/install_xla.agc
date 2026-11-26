## Installs the XLA CPU plugin of `Backend.Xla` (`tensor/DESIGN.md` §7): downloads the plugin of this machine from our
## release of the XLA PJRT build (`Xla.RELEASE`, made by the `XLA plugin` workflow), checks its sha256 and keeps it in
## the ml cache, in a folder named by that sha256. A plugin already there is kept. From `tensor/`:
##
## ```sh
## agentica tools/install_xla.agc
## ```
##
## Prints the plugin's path. `AGENTICA_ML_CACHE` moves the cache (`Xla.cache_dir`).

import HTTP from "std/http"
import Xla from "../src/xla"
import XlaInstall from "../src/xla_install"

def main() -> Void
  case install()
    ok(path) -> puts(path)
    error(e) ->
      IO.warn("install_xla: #{e.message}")
      System.exit(1)
  end
end

private

def install() -> Result[String]
  p = try Xla.plugin()
  path = try Xla.plugin_path()
  return ok(path) if File.stat(path).ok?
  client = try HTTP.client()
  installed = XlaInstall.install(client, p, try Xla.cache_dir())
  HTTP.close_client(client)
  installed
end
