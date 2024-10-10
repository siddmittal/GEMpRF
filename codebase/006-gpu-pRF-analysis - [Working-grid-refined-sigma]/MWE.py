import numpy as np
import cupy as cp

# CUDA custom raw kernel for cuPy
custom_kernel = cp.RawKernel(r'''
extern "C" __global__
	__global__ void custom_kernel(
		double* result_holder,
		double* x_arr,
		double* y_arr,
		double* z_arr,
		double* data_x,
		double* data_y,
		int nRows,
		int nCols,
		int nZ,
		int nDataRow,
		int nDataCol,
		int device_id
	)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int frame = blockIdx.z * blockDim.z + threadIdx.z;

		if (row == 0 && col == 0 && frame == 0)
		{
			printf("Gpu-%d\n", device_id);
            for(int m = 0; m<nZ; m++)
            {
                printf("Z=%f\n", z_arr[m]);
            }
		}

		if (row < nRows && col < nCols && frame < nZ)
		{
			double y_m = y_arr[row];
			double x_m = x_arr[col];
			double z_val = z_arr[frame];

			int meanPairIdx = (frame * (nRows * nCols)) + (row * nCols + col);
			int currentStartIdx = meanPairIdx * (nDataCol * nDataRow);

			int dataIdx = currentStartIdx;
			for (int my_row = 0; my_row < nDataRow; my_row++)
			{
				for (int my_col = 0; my_col < nDataCol; my_col++)
				{
					double y = data_y[my_row];
					double x = data_x[my_col];


					double myPower = -((x - x_m) + (y - y_m) * (y - y_m)) / (z_val * z_val);

					if (dataIdx >= (nRows * nCols * nZ * nDataCol * nDataRow))
					{
						printf("GGpu-%d\n", device_id);
					}

					result_holder[dataIdx] = 1.5; //exp(myPower); 
					dataIdx++;
				}
			}
		}
	}
''', 'custom_kernel')



def compute_chunk(kernel, xx_gpu, yy_gpu, zz_gpu, z_batch_size,  myData_gpu, selected_gpu_idx):    
    # CUDA grid dimensions
    block_dim = (32, 32, 1)
    bx = int((nRows + block_dim[0] - 1) / block_dim[0])
    by = int((nRows + block_dim[1] - 1) / block_dim[1])
    bz = int((nZ + block_dim[2] - 1) / block_dim[2])
    grid_dim = (bx, by, bz)

    #------some required args
    another_xx_gpu = cp.linspace(-9, +9, nCols)
    another_yy_gpu = cp.linspace(-9, +9, nRows)

    # Launch kernels  
    with cp.cuda.Device(selected_gpu_idx):
        result_gc_curves_gpu = cp.zeros((nRows * nCols * z_batch_size * obj_width * obj_height), dtype=cp.float64)

        #---if device is 0
        if selected_gpu_idx == 0:            
            kernel(grid_dim, block_dim, (
            result_gc_curves_gpu,
            another_xx_gpu,
            another_yy_gpu,
            zz_gpu,
            xx_gpu, 
            yy_gpu, 
            nRows, 
            nCols, 
            z_batch_size,
            obj_width,
            obj_height,
            selected_gpu_idx))
        # if other than default device is selected
        else: 
            kernel(grid_dim, block_dim, (
            result_gc_curves_gpu,
            cp.array(another_xx_gpu), # copy array to current device
            cp.array(another_yy_gpu), # copy array to current device
            cp.array(zz_gpu), # copy array to current device
            cp.array(xx_gpu), # copy array to current device
            cp.array(yy_gpu), # copy array to current device
            nRows,
            nCols,
            z_batch_size,
            obj_width,
            obj_height,
            selected_gpu_idx))


        #--------reshape results
        new_nRows = nRows * nCols * z_batch_size
        new_nCols = obj_height * obj_width
        partial_result_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (new_nRows, new_nCols))


        # final
        final_result = cp.dot(partial_result_rowmajor_gpu, myData_gpu)

    return final_result

def dummy_function(kernel):
    zz_gpu =  cp.linspace(0.5, 5, nZ)
    x_range_gpu = cp.linspace(-5, 5, 101)
    y_range_gpu = cp.linspace(-5, 5, 101)
    myData_columnmajor_gpu = cp.random.normal(0.3, 0.9, size=(10201, 300), dtype=cp.float64)

    # results batches
    result_batches = []            

    data_length = myData_columnmajor_gpu.shape[1]
    nDataPerZ = nRows * nCols
    
    # process batches of data
    per_gpu_assigned_z_batch_size = int(nZ / num_gpus)            
    gpu_idx = 0
    for z_idx in range(0, nZ, per_gpu_assigned_z_batch_size):
        result_batch_current_gpu = None              

        with cp.cuda.Device(gpu_idx):
            result_batch_current_gpu = cp.zeros((nRows * nCols * per_gpu_assigned_z_batch_size, data_length), dtype=cp.float64)
            myData_current_device = None                
            if gpu_idx == 0:
                myData_current_device = myData_columnmajor_gpu
            else:
                myData_current_device = cp.array(myData_columnmajor_gpu)
            
            z_arr_gpu = zz_gpu[z_idx : z_idx + per_gpu_assigned_z_batch_size]

            possible_z_batch_size = 2
            selected_gpu_possible_z_chunk_size = per_gpu_assigned_z_batch_size if per_gpu_assigned_z_batch_size <= possible_z_batch_size else possible_z_batch_size                

            # compute chunks of batches                                                 
            for z_chunk_idx in range(0, per_gpu_assigned_z_batch_size, selected_gpu_possible_z_chunk_size):
                from_z_idx = (z_idx * per_gpu_assigned_z_batch_size) +  z_chunk_idx  
                max_chunk_idx = (z_idx + 1)*per_gpu_assigned_z_batch_size
                if (z_chunk_idx + selected_gpu_possible_z_chunk_size) > max_chunk_idx:
                    chunk_size = max_chunk_idx - from_z_idx
                else:
                    chunk_size = selected_gpu_possible_z_chunk_size
                to_z_idx =  (z_idx * per_gpu_assigned_z_batch_size) +  (z_chunk_idx + chunk_size)                                                           
                z_arr_gpu = zz_gpu[from_z_idx : to_z_idx]
                chunk_result_gpu = compute_chunk(kernel, x_range_gpu, y_range_gpu, z_arr_gpu, chunk_size, myData_current_device, gpu_idx)            
                result_batch_current_gpu[z_chunk_idx * nDataPerZ: (z_chunk_idx + chunk_size) * nDataPerZ, :] = chunk_result_gpu    
            
            # append batch to results
            result_batches.append(result_batch_current_gpu)

        # go to next GPU                           
        gpu_idx = gpu_idx + 1 
    
    return result_batches

##################----main()---------###############
# NOTE: Change this to your number of GPUs
num_gpus = 4

# some constants
nRows = 51
nCols = 51
nZ = 8
obj_width = 101
obj_height = 101
S_res = dummy_function(custom_kernel)

print()

