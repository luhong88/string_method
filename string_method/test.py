import multiprocessing

def test_fun(queue, arg):
    try:
        if arg == 10:
            raise KeyError()
        else:
            queue.put(arg**2)
    except Exception as e:
        queue.put(e)

q= multiprocessing.Queue()
processes= []

for i in range(20):
    p= multiprocessing.Process(target= test_fun, args= (q, i))
    processes.append(p)
    p.start()

results= []
for p in processes:
    results.append(q.get())
    print('this is process %s' %p)

for p in processes:
    p.join

print(results)