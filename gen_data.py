import numpy as np
import subprocess
m = 1024
n = 1024
k = 1024

atol=1e-04
rtol=1e-04


'''
a = np.ones(m*k, dtype=np.float16).reshape((m, k))
b = np.ones(k*n, dtype=np.float16).reshape((k, n))

for i in range(m):
    for j in range(k):
        a[i, j] = (k * i + j) / 1024.0

for i in range(k):
    for j in range(n):
        b[i, j] = (n * i + j) / 1024.0

c = np.ones(m * n, dtype=np.float32).reshape((m, n))
for i in range(m):
    for j in range(n):
        c[i, j] = 0
'''

np.random.seed(4)
a = np.random.uniform(low=-100, high=100, size=(m, k)).astype(np.float16)
b = np.random.uniform(low=-100, high=100, size=(k, n)).astype(np.float16)
c = np.random.uniform(low=-100, high=100, size=(m, n)).astype(np.float32)

np.save("a.npy", a)
np.save("b.npy", b)
np.save("c.npy", c)

a_f32 = np.array(a, dtype=np.float32)
b_f32 = np.array(b, dtype=np.float32)
golden = np.matmul(a_f32, b_f32)
golden = np.add(golden, c)

np.save("golden.npy", golden)
np.savetxt("golden.txt", golden)

result = subprocess.run(
        ['/home/peter/github/build.iluvatar.debug/tools/iree-run-module',
         '--device=iluvatar',
         '--input=@a.npy',
         '--input=@b.npy',
         '--input=@c.npy',
         '--module=dump/matmul.vmfb',
         '--output=@res.npy'], capture_output=True, text=True)

print(result)

res = np.load("res.npy")

print("-" * 128)
if np.allclose(golden, res, atol=atol, rtol=rtol):
    printf("Passed")
else:
    diff = np.abs(golden - res)
    not_close_indices = np.where(diff > atol + np.abs(golden) * rtol)
    print(not_close_indices)

    for index in zip(*not_close_indices):
        print("{0}: golden={1}, test={2}\n".format(index, golden[index], res[index]))
