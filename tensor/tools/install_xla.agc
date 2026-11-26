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
  p = try Xla.plugin()
  path = try Xla.plugin_path()
  return ok(path) if File.stat(path).ok?
  try Dir.mkdir(try Xla.cache_dir())
  part = try Path.join([try Xla.cache_dir(), "#{p.sha256}.part-#{System.pid()}"])
  fetched = fetch(p, part)
  if !fetched.ok?
    _ = Path.remove(part)
    return error(fetched.message)
  end

  # Another install may have finished first: its folder holds the same plugin.
  if !Path.rename(part, try Xla.plugin_dir(p)).ok?
    _ = Path.remove(part)
  end

  return error("the plugin is not at #{path} after the install") if !File.stat(path).ok?
  ok(path)
end

## The plugin `p` in the new private folder `dest`, checked against its sha256.
def fetch(p: Xla.Plugin, dest: String) -> Result[Void]
  try Dir.mkdir_private(dest)
  client = try HTTP.client()
  url = resolve(client, p.url)
  if !url.ok?
    HTTP.close_client(client)
    return error(url.message)
  end

  options = HTTP.DownloadOptions(timeout_ms: some(TIMEOUT_MS), sha256: some(p.sha256))
  got = HTTP.download(client, try url, try Path.join([dest, Xla.FILE]), options)
  HTTP.close_client(client)
  saved = try got
  return error("#{p.url} answered #{saved.status}") if saved.status != 200
  ok()
end

## Where the asset at `url` is: `url` itself, or the https address a GitHub release answers with (`HTTP.download` does
## not follow a redirect).
def resolve(client: HTTP.Client, url: String) -> Result[String]
  head = try HTTP.request(client, "HEAD", url, "".bytes)
  return ok(url) if head.status == 200
  return error("#{url} answered #{head.status}") if ![301, 302, 307, 308].include?(head.status)
  where = HTTP.header_values(head.headers, "location")
  return error("#{url} redirects somewhere that is not https") if where.empty? || !where[0].start_with?("https://")
  ok(where[0])
end
