l1 = [int(x) for x in input().split()]


def find_pos_neighbors(alist):
    result = []
    for i in range(1, (len(alist) - 1)):
        if alist[i - 1] * alist[i + 1] >= 0:
            result.append(alist[i])
    return result


print(find_pos_neighbors(l1))


