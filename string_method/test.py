import multiprocessing

def test_fun(queue, arg):
    queue.put(arg**2)

q= multiprocessing.Queue()
processes= []

for i in range(20):
    p= multiprocessing.Process(target= test_fun, args= (q, i))
    processes.append(p)
    p.start()

results= []
for p in processes:
    results.append(q.get())

for p in processes:
    p.join

print(results)