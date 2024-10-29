import numpy as np
import subprocess
m = 128 
n = 128
k = 128

atol=1e-04
rtol=1e-04

np.random.seed(4)

##################################################################
#   Init funcs                                                  #
##################################################################

def SequenceA():
    a = np.ones(m*k, dtype=np.int8).reshape((m, k))
    for i in range(m):
        for j in range(k):
            a[i, j] = (k * i + j) % 128
    return a


def SequenceB():
    b = np.ones(k*n, dtype=np.int8).reshape((k, n))
    for j in range(k):
        for i in range(n):
            b[j, i] = (n * j + i) % 128
    return b

def SequenceC():
    c = np.ones(m * n, dtype=np.int32).reshape((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = (n * i + j) % 128
    return c

def ConstantA(v = 0):
    a = np.ones(m*k, dtype=np.int8).reshape((m, k))
    for i in range(m):
        for j in range(k):
            a[i, j] = v
    return a


def ConstantB(v = 0):
    b = np.ones(k*n, dtype=np.int8).reshape((k, n))
    for j in range(k):
        for i in range(n):
            b[j, i] = v
    return b

def ConstantC(v = 0):
    c = np.ones(m * n, dtype=np.int32).reshape((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = v
    return c

def IdentityA():
    a = np.ones(m*k, dtype=np.int8).reshape((m, k))
    for i in range(m):
        for j in range(k):
            if i == j:
                a[i, j] = 1
            else:
                a[i, j] = 0
    return a

def IdentityB():
    b = np.ones(k*n, dtype=np.int8).reshape((k, n))
    for i in range(k):
        for j in range(n):
            if i == j:
                b[i, j] = 1
            else:
                b[i, j] = 0
    return b

def RandomA():
    a = np.random.uniform(low=0, high=128, size=(m, k)).astype(np.int8)
    return a

def RandomB():
    b = np.random.uniform(low=0, high=128, size=(k, n)).astype(np.int8)
    return b

def RandomC():
    c = np.random.uniform(low=0, high=128, size=(m, n)).astype(np.int32)
    return c

#a = np.ones(m*k, dtype=np.int8).reshape((m, k))
#b = np.ones(k*n, dtype=np.int8).reshape((k, n))
#c = np.ones(m * n, dtype=np.int32).reshape((m, n))


#
# prepare operands a, b and c
#
a = ConstantA(1)
b = IdentityB()
c = ConstantC(0)

#
# dump operands a, b and c
#
np.save("a.npy", a)
np.save("b.npy", b)
np.save("c.npy", c)

#
# generate golden and dump
#
a_f32 = np.array(a, dtype=np.int32)
b_f32 = np.array(b, dtype=np.int32)
golden = np.matmul(a_f32, b_f32)
golden = np.add(golden, c)

np.save("golden.npy", golden)
np.savetxt("golden.txt", golden, fmt="%3.f")

#
# iree run module and dump result
#
result = subprocess.run(
        ['/home/peter/github/build.iluvatar.debug/tools/iree-run-module',
         '--device=iluvatar',
         '--input=@a.npy',
         '--input=@b.npy',
         '--input=@c.npy',
         '--module=dump/matmul-i8.vmfb',
         '--output=@res.npy'], capture_output=True, text=True)

print(result)

res = np.load("res.npy")
np.savetxt("res.txt", golden, fmt="%3.f")
np.savetxt("a.txt", a, fmt="%3.f")
np.savetxt("b.txt", b, fmt="%3.f")

#
# compare result with golden and dump compare result to console
#
print("-" * 128)
if np.allclose(golden, res, atol=atol, rtol=rtol):
    print("Passed")
else:
    diff = np.abs(golden - res)
    not_close_indices = np.where(diff > atol + np.abs(golden) * rtol)
    print(not_close_indices)
    #for index in zip(*not_close_indices):
    #    print("{0}: golden={1}, test={2}\n".format(index, golden[index], res[index]))


#
# compare specific sub matrix and dump
#
rstart = 0
cstart = 0
rend = 64
cend = 64
#print(res[rstart:rend,cstart:cend] == golden[rstart:rend,cstart:cend])
np.savetxt("res.{0}-{1}.{2}-{3}.txt".format(rstart,rend,cstart,cend), res[rstart:rend,cstart:cend], fmt="%3.f")
np.savetxt("golden.{0}-{1}.{2}-{3}.txt".format(rstart,rend,cstart,cend), golden[rstart:rend,cstart:cend], fmt="%3.f")


# {'edgeitems': 128, 'threshold': 128, 'floatmode': 'maxprec', 'precision': 8, 'suppress': False, 'linewidth': 128, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-', 'formatter': None, 'legacy': False}
np.set_printoptions(linewidth=128)
np.set_printoptions(threshold=np.inf)


#
# Verify if a == res, while no mmad, just copy a first half to res
#

for i in range(m):
    base = int(i / 16) * 16
    rem = int(i % 16)
    t = base + int(rem % 4) * 4 + int(rem / 4)
    #print("a{0}->res{1}".format(i, t))
    print("a[{0}] vs res[{1}]: {2}, {3}, {4}".format(i, t, (a[i,96:112] == res[t,:16]), a[i,96:112], res[t,:16]))

