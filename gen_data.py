import numpy as np
import subprocess
m = 512 
n = 512
k = 512

'''
ABType = np.int8
CType = np.int32
VMFile = 'dump-i8/matmul.vmfb'
Divisor = 128
'''
ABType = np.float16
CType = np.float16
VMFile = 'dump-f16/matmul.vmfb'
Divisor = 100000


atol=1e-03
rtol=1e-04

np.random.seed(4)

##################################################################
#   Init funcs                                                  #
##################################################################

def SequenceA():
    a = np.ones(m*k, dtype=ABType).reshape((m, k))
    for i in range(m):
        for j in range(k):
            a[i, j] = (k * i + j) / Divisor
    return a


def SequenceB():
    b = np.ones(k*n, dtype=ABType).reshape((k, n))
    for j in range(k):
        for i in range(n):
            b[j, i] = (n * j + i) / Divisor
    return b

def SequenceC():
    c = np.ones(m * n, dtype=CType).reshape((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = (n * i + j) / Divisor
    return c

def ConstantA(v = 0):
    a = np.ones(m*k, dtype=ABType).reshape((m, k))
    for i in range(m):
        for j in range(k):
            a[i, j] = v
    return a


def ConstantB(v = 0):
    b = np.ones(k*n, dtype=ABType).reshape((k, n))
    for j in range(k):
        for i in range(n):
            b[j, i] = v
    return b

def ConstantC(v = 0):
    c = np.ones(m * n, dtype=CType).reshape((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = v
    return c

def IdentityA():
    a = np.ones(m*k, dtype=ABType).reshape((m, k))
    for i in range(m):
        for j in range(k):
            if i == j:
                a[i, j] = 1
            else:
                a[i, j] = 0
    return a

def IdentityB():
    b = np.ones(k*n, dtype=ABType).reshape((k, n))
    for i in range(k):
        for j in range(n):
            if i == j:
                b[i, j] = 1
            else:
                b[i, j] = 0
    return b

def RandomA():
    a = np.random.uniform(low=-3, high=3, size=(m, k)).astype(ABType)
    return a

def RandomB():
    b = np.random.uniform(low=-3, high=3, size=(k, n)).astype(ABType)
    return b

def RandomC():
    c = np.random.uniform(low=-3, high=3, size=(m, n)).astype(CType)
    return c

#a = np.ones(m*k, dtype=ABType).reshape((m, k))
#b = np.ones(k*n, dtype=ABType).reshape((k, n))
#c = np.ones(m * n, dtype=CType).reshape((m, n))


#
# prepare operands a, b and c
#
a = RandomA()
b = RandomB()
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
a_f32 = np.array(a, dtype=CType)
b_f32 = np.array(b, dtype=CType)
golden = np.matmul(a_f32, b_f32)
golden = np.add(golden, c)

np.save("golden.npy", golden)
np.savetxt("golden.txt", golden, fmt="%3.5f")

#
# iree run module and dump result
#
result = subprocess.run(
        ['/home/peter/github/build.iluvatar.debug/tools/iree-run-module',
         '--device=iluvatar',
         '--input=@a.npy',
         '--input=@b.npy',
         #'--input=@c.npy',
         '--module={0}'.format(VMFile),
         '--output=@res.npy'], capture_output=True, text=True)

print(result)

res = np.load("res.npy")
np.savetxt("res.txt", res, fmt="%3.5f")
np.savetxt("a.txt", a, fmt="%3.5f")
np.savetxt("b.txt", b, fmt="%3.5f")
np.savetxt("c.txt", c, fmt="%3.5f")

#
# compare result with golden and dump compare result to console
#
print("-" * 128)
if np.allclose(golden, res, atol=atol, rtol=rtol):
    print("Passed")
else:
    diff = np.abs(golden - res)
    not_close_indices = np.where(diff > atol + np.abs(golden) * rtol)
    print(len(not_close_indices[0]))
    #for index in zip(*not_close_indices):
    #    print("{0}: golden={1}, test={2}\n".format(index, golden[index], res[index]))


# {'edgeitems': 128, 'threshold': 128, 'floatmode': 'maxprec', 'precision': 8, 'suppress': False, 'linewidth': 128, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-', 'formatter': None, 'legacy': False}
np.set_printoptions(linewidth=128)
np.set_printoptions(threshold=np.inf)


'''
#
# Verify A of b8 row major in global memory copied to C via SME instruction
#
for i in range(m):
    base = i // 16 * 16
    ii = (i - base) // 4 + (i - base) % 4 * 4 + base
    aa = a[i,-32:-16]
    for d in range(128 // 16):
        rr = res[ii,16*d:16*d +16]
        #print("{0} vs {1}".format(i, ii))
        print(aa == rr)

#
# Verify B of b8 row major in global memory copied to C via SME instruction
#
for i in range(32):
    base = i // 16 * 16
    ii = (i - base) // 4 + (i - base) % 4 * 4
    aa = b[i % 16 - 32,:16]
    rr = res[base+ ii,:16]
    #print("{0} vs {1}".format(ii - 32, ii + base))
    #print("{0} vs {1}".format(i, ii + base))
    print("b[{0}] vs res[{1}] : {2}".format(i-32, ii + base, aa == rr))
'''

