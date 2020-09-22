def traverse(res, funcs, i):
    if i == len(funcs):
        return res
    else:
        new_res = funcs[i](res)
        return traverse(new_res, funcs, i + 1)


def makeFunc(n):
    def add(x):
        return 2 + x

    return add


l_funcs = []

for i in range(4):
    l_funcs.append(makeFunc(i))

res = traverse(0, l_funcs, 0)

print(res)
