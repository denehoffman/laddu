import laddu

with laddu.mpi.MPI():
    print(f'{laddu.mpi.get_rank()}')
