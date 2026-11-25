import time, numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
print(jax.devices())
mesh = Mesh(np.array(jax.devices()[:2]), ("m",))
x = np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 13 / 13
w = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 7 / 7
want = x @ w
for name, xs, ws, outs in [("col", P(), P(None, "m"), P(None, "m")), ("row", P(None, "m"), P("m", None), P())]:
    f = jax.jit(lambda a, b: a @ b, in_shardings=(NamedSharding(mesh, xs), NamedSharding(mesh, ws)), out_shardings=NamedSharding(mesh, outs))
    low = f.lower(x, w)
    if name == "col":
        print(low.as_text())
    t = time.time(); c = low.compile(); ct = time.time() - t
    xa, wa = jax.device_put(x, NamedSharding(mesh, xs)), jax.device_put(w, NamedSharding(mesh, ws))
    c(xa, wa).block_until_ready()
    ts = []
    for _ in range(31):
        t = time.time(); c(xa, wa).block_until_ready(); ts.append(time.time() - t)
    got = np.asarray(c(xa, wa))
    print(name, "compile %.3f s run %.3f ms gap %.2g" % (ct, sorted(ts)[15] * 1000, np.abs(got - want).max()))
