## The download of `tools/install_xla.agc` against a server on localhost (port 0) that answers what a GitHub release
## does: the file, a redirect and a missing asset. Nothing leaves the machine.

import HTTP from "std/http"
import Xla from "../src/xla"
import XlaInstall from "../src/xla_install"

BODY = "xla plugin bytes"
# `printf 'xla plugin bytes' | sha256sum`
DIGEST = "3e7e977cf7e1b0c1c77a505f304aaa58508b42a15bb1f5a85b61a821075e6009"
WRONG_DIGEST = "0000000000000000000000000000000000000000000000000000000000000000"

private

def answer(status: Int, headers: List[HTTP.Header], body: String) -> Result[HTTP.Response]
  ok(HTTP.Response(status: status, headers: headers, body: body.bytes))
end

def handle(request: HTTP.Request) -> Result[HTTP.Response]
  return answer(200, [], BODY) if request.path == "/asset"
  if request.path == "/to-http"
    return answer(302, [HTTP.Header(name: "Location", value: "http://127.0.0.1:1/asset")], "")
  end

  if request.path == "/to-https"
    return answer(302, [HTTP.Header(name: "Location", value: "https://releases.invalid/asset")], "")
  end

  answer(404, [], "not found")
end

def plugin(base: String, path: String, sha256: String) -> Xla.Plugin
  Xla.Plugin(arch: "x86_64", url: base + path, size: BODY.size, sha256: sha256)
end

## Runs `body` with a client, the server's address and an empty cache folder, then removes the folder. The server and
## its connection tasks have ended when this returns, whatever `body` did: a test that returns with a task running
## fails with `RUNTIME-TEST-PENDING-TASKS`.
def with_server(body: Fn[HTTP.Client, String, String, Result[Void]]) -> Result[Void]
  server = try HTTP.serve("127.0.0.1", 0, HTTP.app(&handle))
  result = with_client(HTTP.local_port(server), body)
  HTTP.close_server(server)
  _ = HTTP.wait(server)
  result
end

def with_client(port: Int, body: Fn[HTTP.Client, String, String, Result[Void]]) -> Result[Void]
  client = try HTTP.client()
  result = with_cache(client, "http://127.0.0.1:#{port}", body)
  HTTP.close_client(client)
  result
end

def with_cache(client: HTTP.Client, base: String, body: Fn[HTTP.Client, String, String, Result[Void]]) -> Result[Void]
  cache = "/tmp/agentica-xla-install-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(cache, 5000)
  result = body.call(client, base, cache)
  _ = Path.remove(cache, 5000)
  result
end

def test_a_file_with_the_pinned_digest_is_installed_by_its_sha256() -> Result[Void]
  with_server do |client: HTTP.Client, base: String, cache: String|
    path = try XlaInstall.install(client, plugin(base, "/asset", DIGEST), cache)
    try Test.eq(path, "#{cache}/#{DIGEST}/#{Xla.FILE}")
    try Test.eq(try File.read_text(path), BODY)
    # Installed again, it is kept: nothing is asked of the server.
    Test.eq(try XlaInstall.install(client, plugin("http://127.0.0.1:1", "/asset", DIGEST), cache), path)
  end
end

def test_a_file_with_another_digest_is_an_error_and_leaves_no_folder() -> Result[Void]
  with_server do |client: HTTP.Client, base: String, cache: String|
    r = XlaInstall.install(client, plugin(base, "/asset", WRONG_DIGEST), cache)
    try Test.assert(!r.ok?, "a file with the wrong sha256 was installed")
    try Test.assert(r.message.include?(base + "/asset"), r.message)
    Test.eq(try Dir.list(cache), [])
  end
end

def test_a_redirect_that_is_not_https_is_refused() -> Result[Void]
  with_server do |client: HTTP.Client, base: String, cache: String|
    r = XlaInstall.install(client, plugin(base, "/to-http", DIGEST), cache)
    try Test.assert(!r.ok?, "a redirect to http was followed")
    try Test.assert(r.message.include?("not https"), r.message)
    Test.eq(try Dir.list(cache), [])
  end
end

def test_an_https_redirect_is_the_address_to_download() -> Result[Void]
  with_server do |client: HTTP.Client, base: String, _cache: String|
    Test.eq(try XlaInstall.resolve(client, base + "/to-https"), "https://releases.invalid/asset")
  end
end

def test_a_status_that_is_not_200_names_the_url() -> Result[Void]
  with_server do |client: HTTP.Client, base: String, cache: String|
    r = XlaInstall.install(client, plugin(base, "/gone", DIGEST), cache)
    try Test.assert(!r.ok?, "a 404 was installed")
    try Test.assert(r.message.include?(base + "/gone"), r.message)
    try Test.assert(r.message.include?("404"), r.message)
    Test.eq(try Dir.list(cache), [])
  end
end
