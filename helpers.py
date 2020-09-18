def numLayers(size):
    i = 0
    return divideByTwo(size, i)


def divideByTwo(n, i):
    print("Dividing")
    if n == 1:
        return i
    else:
        i += 1
        return divideByTwo(n / 2, i)
