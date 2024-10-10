import cupy as cp

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def print_gpu_memory_stats():    
    total_mempool = mempool.total_bytes() / (1024 * 1024)
    used_mempool = mempool.used_bytes() / (1024 * 1024)
    used_pinned_mempool = pinned_mempool.n_free_blocks()    
    print(f'Total Mempool Size: {total_mempool} MB')
    print(f'Used Mempool Size: {used_mempool} MB')
    print(f'Free pinned mempool blocks: {used_pinned_mempool} blocks')

def foo():
    gc_gpu = cp.zeros((51 * 51 * 8 * 101 * 101), dtype=cp.float64)
    dgc_gpu = cp.ones((51 * 51 * 8 * 101 * 101), dtype=cp.float64)

    print_gpu_memory_stats()


    result = gc_gpu + dgc_gpu

    # del gc_gpu
    # del dgc_gpu

    test = 0

    anotherTest = 5

    # free GPU memory ---- MEMORY IS NOT REALEASED UNTIL THE LINES BELOW ARE EXECUTED !!!
    # mempool = cp.get_default_memory_pool()
    # mempool.free_all_blocks()

    new_result = result * 2

    print_gpu_memory_stats() 

    return new_result
    # return 0

def boo():
    test = 'just another test function'

####--------program---------#####
print_gpu_memory_stats()

foo = foo()

print_gpu_memory_stats() 


boo()
testVar = "GPU memory is still not released"

# free GPU memory
# mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

print_gpu_memory_stats()


# just to keep the program a little longer
message = "GPU memory gets released on the execution of the line above"
