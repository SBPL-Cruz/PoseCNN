#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "backprojecting_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void BackprojectForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_label, const Dtype* bottom_depth,
    const Dtype* bottom_meta_data, const Dtype* bottom_label_3d,
    const int height, const int width, const int channels, const int num_classes, const int num_meta_data,
    const int grid_size, const int kernel_size, const float threshold, Dtype* top_data, Dtype* top_label, float* top_flag)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
    // (n, d, h, w, c) is an element in the output
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % grid_size;
    n /= grid_size;
    int h = n % grid_size;
    n /= grid_size;
    int d = n % grid_size;
    n /= grid_size;

    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]

    // voxel location in 3D
    const Dtype* meta_data = bottom_meta_data + n * num_meta_data;
    Dtype X = d * meta_data[42] + meta_data[45];
    Dtype Y = h * meta_data[43] + meta_data[46];
    Dtype Z = w * meta_data[44] + meta_data[47];

    // apply pose_world2live
    Dtype X1 = meta_data[18] * X + meta_data[19] * Y + meta_data[20] * Z + meta_data[21];
    Dtype Y1 = meta_data[22] * X + meta_data[23] * Y + meta_data[24] * Z + meta_data[25];
    Dtype Z1 = meta_data[26] * X + meta_data[27] * Y + meta_data[28] * Z + meta_data[29];

    // apply the intrinsic matrix
    Dtype x1 = meta_data[0] * X1 + meta_data[1] * Y1 + meta_data[2] * Z1;
    Dtype x2 = meta_data[3] * X1 + meta_data[4] * Y1 + meta_data[5] * Z1;
    Dtype x3 = meta_data[6] * X1 + meta_data[7] * Y1 + meta_data[8] * Z1;
    int px = round(x1 / x3);
    int py = round(x2 / x3);

    // initialization
    top_data[index] = 0;
    if (c == 0)
    {
      for(int cl = 0; cl < num_classes; cl++)
        top_label[(n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) * num_classes + cl] = 0;
    }

    // check a neighborhood around (px, py)
    int count = 0;
    for (int x = px - kernel_size; x <= px + kernel_size; x++)
    {
      for (int y = py - kernel_size; y <= py + kernel_size; y++)
      {
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
          int index_pixel = n * height * width + y * width + x;
          Dtype depth = bottom_depth[index_pixel];

          // distance of this voxel to camera center
          Dtype dvoxel = Z1;

          // check if the voxel is on the surface
          if (fabs(depth - dvoxel) < threshold)
          {
            count++;
            // data
            top_data[index] += bottom_data[index_pixel * channels + c];
            // label
            if (c == 0)
            {
              for(int cl = 0; cl < num_classes; cl++)
                top_label[(n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) * num_classes + cl] += bottom_label[index_pixel * num_classes + cl];
            }
          }
        }
      }
    }

    if (count == 0)
    {
      // flag
      top_flag[index] = 0;
      // label
      if (c == 0)
      {
        for(int cl = 0; cl < num_classes; cl++)
          top_label[(n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) * num_classes + cl] =
            bottom_label_3d[(n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) * num_classes + cl];
      }
    }
    else
    {
      // data
      top_data[index] /= count;
      // flag
      top_flag[index] = 1;
      // label
      if (c == 0)
      {
        for(int cl = 0; cl < num_classes; cl++)
          top_label[(n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) * num_classes + cl] /= count;
      }
    }
  }
}

// bottom_data: (batch_size, height, width, channels)
bool BackprojectForwardLaucher(
    const float* bottom_data, const float* bottom_label,
    const float* bottom_depth, const float* bottom_meta_data, const float* bottom_label_3d,
    const int batch_size, const int height, const int width,
    const int channels, const int num_classes, const int num_meta_data,
    const int grid_size, const int kernel_size, const float threshold,
    float* top_data, float* top_label, float* top_flag, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 512;
  cudaError_t err;

  const int output_size = batch_size * grid_size * grid_size * grid_size * channels;
  BackprojectForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_label, bottom_depth, bottom_meta_data, bottom_label_3d, height, width, channels, num_classes, num_meta_data,
      grid_size, kernel_size, threshold, top_data, top_label, top_flag);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void BackprojectBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_depth, const Dtype* bottom_meta_data,
    const int height, const int width, const int channels, const int num_meta_data,
    const int grid_size, Dtype* bottom_diff)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
    // (n, h, w, c) coords in bottom data
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    int index_pixel = n * height * width + h * width + w;
    Dtype depth = bottom_depth[index_pixel];

    // find the voxel for this pixel

    // backproject the pixel to 3D
    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Dtype* meta_data = bottom_meta_data + n * num_meta_data;

    // apply the inverse intrinsic matrix
    int offset = 9;
    Dtype RX = meta_data[offset + 0] * w + meta_data[offset + 1] * h + meta_data[offset + 2];
    Dtype RY = meta_data[offset + 3] * w + meta_data[offset + 4] * h + meta_data[offset + 5];
    Dtype RZ = meta_data[offset + 6] * w + meta_data[offset + 7] * h + meta_data[offset + 8];

    // compute the 3D points in camera's coordinate system
    Dtype X = depth * RX;
    Dtype Y = depth * RY;
    Dtype Z = depth * RZ;

    // apply pose_live2world
    Dtype X1 = meta_data[30] * X + meta_data[31] * Y + meta_data[32] * Z + meta_data[33];
    Dtype Y1 = meta_data[34] * X + meta_data[35] * Y + meta_data[36] * Z + meta_data[37];
    Dtype Z1 = meta_data[38] * X + meta_data[39] * Y + meta_data[40] * Z + meta_data[41];

    // voxel location in 3D
    int vd = round((X1 - meta_data[45]) / meta_data[42]);
    int vh = round((Y1 - meta_data[46]) / meta_data[43]);
    int vw = round((Z1 - meta_data[47]) / meta_data[44]);

    // get the gradient
    if (vd >= 0 && vd < grid_size && vh >= 0 && vh < grid_size && vw >= 0 && vw < grid_size)
      bottom_diff[index] = top_diff[(n * grid_size * grid_size * grid_size + vd * grid_size * grid_size + vh * grid_size + vw) * channels + c];
    else
      bottom_diff[index] = 0;
  }
}


bool BackprojectBackwardLaucher(const float* top_diff, const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, const int grid_size,
    float* bottom_diff, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 512;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  BackprojectBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, bottom_depth, bottom_meta_data,
      height, width, channels, num_meta_data, grid_size, bottom_diff);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
