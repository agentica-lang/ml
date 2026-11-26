## The download of the XLA plugin that `tools/install_xla.agc` runs (`tensor/DESIGN.md` §7), apart from the tool so a test can
## call it with a client and a cache folder of its own.

import HTTP from "std/http"
import Xla from "xla"

TIMEOUT_MS = 1800000

## The plugin `p` in the folder `cache` named by its sha256, and the path of its file. A plugin already there is kept.
def install(client: HTTP.Client, p: Xla.Plugin, cache: String) -> Result[String]
  dir = try Path.join([cache, p.sha256])
  path = try Path.join([dir, Xla.FILE])
  return ok(path) if File.stat(path).ok?
  try Dir.mkdir(cache)
  part = try Path.join([cache, "#{p.sha256}.part-#{System.pid()}"])
  fetched = fetch(client, p, part)
  if !fetched.ok?
    _ = Path.remove(part)
    return error(fetched.message)
  end

  # Another install may have finished first: its folder holds the same plugin.
  if !Path.rename(part, dir).ok?
    _ = Path.remove(part)
  end

  return error("the plugin is not at #{path} after the install") if !File.stat(path).ok?
  ok(path)
end

## The plugin `p` in the new private folder `dest`, checked against its sha256.
def fetch(client: HTTP.Client, p: Xla.Plugin, dest: String) -> Result[Void]
  try Dir.mkdir_private(dest)
  url = try resolve(client, p.url)
  options = HTTP.DownloadOptions(timeout_ms: some(TIMEOUT_MS), max_total_bytes: some(p.size), sha256: some(p.sha256))
  got = HTTP.download(client, url, try Path.join([dest, Xla.FILE]), options)
  # A status that is not 200 has a body that does not have the digest, so it fails here too.
  return error("#{p.url}: #{got.message}") if !got.ok?
  ok()
end

## Where the asset at `url` is: `url` itself, or the https address a GitHub release answers with (`HTTP.download` does
## not follow a redirect).
def resolve(client: HTTP.Client, url: String) -> Result[String]
  head = try HTTP.request(client, "HEAD", url, "".bytes)
  return ok(url) if head.status == 200
  return error("#{url} answered #{head.status}") if ![301, 302, 307, 308].include?(head.status)
  where = HTTP.header_values(head.headers, "location")
  # The trust anchor is the sha256 of the file, not the host it comes from; https only keeps the transfer private.
  return error("#{url} redirects somewhere that is not https") if where.empty? || !where[0].start_with?("https://")
  ok(where[0])
end
