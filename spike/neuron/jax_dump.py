import numpy as np, jax
from jax._src import compiler
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
orig = compiler.backend_compile_and_load
def spy(backend, module, executable_devices, options, host_callbacks):
    open("jax_col.mlir", "w").write(module.operation.get_asm(enable_debug_info=False))
    open("jax_opts2.bin", "wb").write(options.SerializeAsString())
    print("build options:", options.executable_build_options)
    return orig(backend, module, executable_devices, options, host_callbacks)
compiler.backend_compile_and_load = spy
mesh = Mesh(np.array(jax.devices()[:2]), ("m",))
x = np.ones((8, 16), np.float32); w = np.ones((16, 32), np.float32)
f = jax.jit(lambda a, b: a @ b, in_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "m"))), out_shardings=NamedSharding(mesh, P(None, "m")))
f(x, w).block_until_ready()
