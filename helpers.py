def getNumLayers(size):
    i = 0
    return divideByTwo(size, i)


def divideByTwo(n, i):
    if n == 1:
        return i
    else:
        i += 1
        return divideByTwo(n / 2, i)
