import numpy as np

a = [1,2,3,2,3,2,3,2,1]
common = np.argmax(a)
print(np.argmax(a))

b =[]
for num in a:
    if num != common:
        b.append(num)
print(b)
print(np.argmax(b))