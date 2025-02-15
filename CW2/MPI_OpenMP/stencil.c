#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include "omp.h"
#include <string.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0
#define THREADS 2

void stencil(const int local_ncols, const int local_nrows, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);
void halo_exchange(float* sendbuf, float* recvbuf, float* section, int left, int right, 
                int local_ncols, int local_nrows, int size, int rank, MPI_Status status);
int calc_ncols_from_rank(int rank, int size, int ny);

int main(int argc, char* argv[])
{
  // Check usage
  if (argc != 4) 
  {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Adding MPI stuff
  int rank;               /* 'rank' of process among it's cohort */
  int size;               /* size of cohort, i.e. num processes started */

  MPI_Status status;     /* struct used by MPI_Recv */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  /* initialise our MPI environment */
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /*
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  local_nrows = nx;
  local_ncols = calc_ncols_from_rank(rank, size, ny);

  sendbuf = (float*) malloc(sizeof(float) * (local_nrows + 2));
  recvbuf = (float*) malloc(sizeof(float) * (local_nrows + 2));

  /* check whether the initialisation was successful */
  if ( local_ncols < 1 )
  {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  int section_ncols = local_ncols + 2;

  float* section = malloc(sizeof(float) * (local_nrows + 2) * section_ncols);
  float* tmp_section = malloc(sizeof(float) * (local_nrows + 2) * section_ncols);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  omp_set_num_threads(THREADS);

  // Initialising the sections
  for(int i = 0; i < local_nrows + 2; i++) 
  {
    for(int j = 0; j < local_ncols + 2; j++) 
    {
      if (j > 0 && j < (local_ncols + 1) && i > 0 && i < (local_nrows + 1))
      {
        section[i * (local_ncols + 2) + j] = image[( i * width + j + (ny/size * rank) )];
        tmp_section[i * (local_ncols + 2) + j] = image[(ny/size * rank + i) * ((local_ncols + 2) + j)];                 
      }
      else
      {
        section[i * (local_ncols + 2) + j] = 0.0f;
        tmp_section[i * (local_ncols + 2) + j] = 0.0f;
      }
    }
  }

  // Call the stencil kernel
  double tic = wtime();
  for(int t = 0; t < niters; ++t) 
  {
    // Halo Exchange from left to right followed by right to left for section
    halo_exchange(sendbuf, recvbuf, section, left, right, local_ncols, local_nrows, size, rank, status);

    // Call stencil from section to tmp_section
    stencil(local_ncols, local_nrows, width, height, section, tmp_section);

    // Halo Exchange from left to right followed by right to left for tmp_section
    halo_exchange(sendbuf, recvbuf, tmp_section, left, right, local_ncols, local_nrows, size, rank, status);

    // Call stencil from tmp_section to section
    stencil(local_ncols, local_nrows, width, height, tmp_section, section);
  }
  double toc = wtime();

  // Gathering
  for(int i = 1; i < local_nrows + 1; i++)
  {
    if(rank == MASTER)
    {
      for(int j = 1; j < local_ncols + 1 ; j++)
      {
        image[(i * width) + j] = section[i * (local_ncols + 2) + j];
      }

      for(int r = 1; r < size; r++)
      {
        int ncols = calc_ncols_from_rank(r, size, ny);
        int offset = r * (ny / size) + 1;       // offset for each rank when storing back to image

        MPI_Recv(&image[(i * width) + offset], ncols, MPI_FLOAT, r, 0, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      MPI_Send(&section[i * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
    }
  }

  // Output if rank is MASTER
  if(rank == MASTER)
  {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }

  MPI_Finalize();

  free(image);
  free(tmp_image);
  free(sendbuf);
  free(recvbuf);
  free(section);
  free(tmp_section);
}

void halo_exchange(float* sendbuf, float* recvbuf, float* section, int left, int right, int local_ncols, int local_nrows, int size, int rank, MPI_Status status)
{
    // Packing the send buffer with the left column
    for(int i = 0; i < local_nrows + 2; ++i)
    {
      sendbuf[i] = section[i * (local_ncols + 2) + 1];
    }

    // Exchanging: Send to the Left, Receive to the Right
    MPI_Sendrecv(sendbuf, local_nrows + 2, MPI_FLOAT, left, 0,
    recvbuf, local_nrows + 2, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);

    // Unpacking the values from the receive buffer into the section
    if(rank != size - 1)
    {
      for(int i = 0; i < local_nrows + 2; ++i)
      {
        section[i * (local_ncols + 2) + local_ncols + 1] = recvbuf[i];
      }
    }

    // Packing the send buffer with the right column
    for(int i = 0; i < local_nrows + 2; ++i)
    {
      sendbuf[i] = section[i * (local_ncols + 2) + local_ncols];
    }     

    // Exchanging: Send to the Right, Receive to the Left
    MPI_Sendrecv(sendbuf, local_nrows + 2, MPI_FLOAT, right, 0,
    recvbuf, local_nrows + 2, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);

    // Unpacking the values from the receive buffer into the section
    if(rank != MASTER)
    {
      for(int i = 0; i < local_nrows + 2; ++i)
      {
        section[i * (local_ncols + 2)] = recvbuf[i];
      }
    }
}

void stencil(const int local_ncols, const int local_nrows, const int width, const int height,
             float* image, float* tmp_image)
{ 
  #pragma omp simd collapse(2)
  for (int i = 1; i < local_nrows + 1; ++i)
  {
    for (int j = 1; j < local_ncols + 1; ++j)
    {
      int cell = j + i * (local_ncols + 2);
      tmp_image[cell] = ((image[cell] * 6.0f) + (image[cell - (local_ncols + 2)] + image[cell + (local_ncols + 2)] + image[cell - 1] +  image[cell + 1]))/10.0f;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int calc_ncols_from_rank(int rank, int size, int ny)
{
  int ncols;

  ncols = ny / size;       /* integer division */
  if ((ny % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += ny % size;  /* add remainder to last rank */
  }

  return ncols;
}
