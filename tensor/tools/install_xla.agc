## Installs the XLA CPU plugin of `Backend.Xla` (`tensor/DESIGN.md` §7): downloads the pinned `xla-cpu-pjrt` wheel of
## this machine from PyPI (`Xla.wheels`), checks its sha256, and extracts the plugin with `unzip` into the ml cache,
## under a folder named by that sha256. A plugin already there is kept. From `tensor/`:
##
## ```sh
## agentica tools/install_xla.agc
## ```
##
## Prints the plugin's path. `AGENTICA_ML_CACHE` moves the cache (`Xla.cache_dir`). `std/archive` does not read the
## wheel: its local headers are ZIP64.

import HTTP from "std/http"
import Xla from "../src/xla"

TIMEOUT_MS = 1800000

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
  w = try Xla.wheel()
  plugin = try Xla.plugin_path()
  return ok(plugin) if File.stat(plugin).ok?
  cache = try Xla.cache_dir()
  try Dir.mkdir(cache)
  whl = try Path.join([cache, "#{w.sha256}-#{System.pid()}.whl"])
  fetched = fetch(w, whl)
  if !fetched.ok?
    _ = Path.remove(whl)
    return error(fetched.message)
  end

  part = try Path.join([cache, "#{w.sha256}.part-#{System.pid()}"])
  extracted = unzip(whl, part)
  _ = Path.remove(whl)
  if !extracted.ok?
    _ = Path.remove(part)
    return error("cannot extract #{Xla.PLUGIN} with unzip: #{extracted.message}")
  end

  # Another install may have finished first: its folder holds the same plugin.
  if !Path.rename(part, try Xla.wheel_dir(w)).ok?
    _ = Path.remove(part)
  end

  return error("the plugin is not at #{plugin} after the install") if !File.stat(plugin).ok?
  ok(plugin)
end

## The wheel `w` at `path`, checked against its sha256.
def fetch(w: Xla.Wheel, path: String) -> Result[Void]
  client = try HTTP.client()
  options = HTTP.DownloadOptions(timeout_ms: some(TIMEOUT_MS), sha256: some(w.sha256))
  got = HTTP.download(client, w.url, path, options)
  HTTP.close_client(client)
  saved = try got
  return error("#{w.url} answered #{saved.status}") if saved.status != 200
  ok()
end

## The plugin of the wheel `whl` extracted into the new folder `dest`.
def unzip(whl: String, dest: String) -> Result[Void]
  try Dir.mkdir_private(dest)
  p = try Process.spawn_captured("unzip", ["-q", whl, Xla.PLUGIN, "-d", dest])
  out = Process.output(p, TIMEOUT_MS)
  code = Process.wait(p, TIMEOUT_MS)
  Process.close(p)
  case code
    ok(0) -> ok()
    ok(_) -> error((try out).stderr.strip)
    error(e) -> error(e.message)
  end
end
