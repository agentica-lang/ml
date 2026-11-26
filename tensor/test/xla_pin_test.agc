## The pinned release of the XLA plugin is the one `tensor/xla-commit` names, and every plugin comes from it.
## It needs no plugin and no native module.

import Xla from "../src/xla"

private

def test_the_release_is_named_by_the_pinned_commit() -> Result[Void]
  commit = (try File.read_text("xla-commit")).strip
  try Test.assert(commit.length == 40, commit)
  Test.assert(Xla.RELEASE == "xla-#{commit.slice(0, 12)}", Xla.RELEASE)
end

def test_every_plugin_is_an_asset_of_the_release() -> Result[Void]
  try Test.assert(Xla.plugins().length == 2, "two plugins")
  for p in Xla.plugins()
    try Test.assert(p.url.include?("/releases/download/#{Xla.RELEASE}/"), p.url)
    try Test.assert(p.sha256.length == 64, p.sha256)
  end

  ok()
end
