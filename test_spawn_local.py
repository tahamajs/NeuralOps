import multiprocessing as mp
import sys

def worker():
    import neuralops.benchmarks.executor
    print('Executor imported!')

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=worker)
    p.start()
    p.join()
    print('Exitcode:', p.exitcode)
