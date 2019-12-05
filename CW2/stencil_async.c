#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include "omp.h"
#include <string.h>
#include <math.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int local_nrows, const int local_ncols,
             float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float * restrict image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float * restrict image);
double wtime(void);
void halo_exchange(float * restrict section, int up, int down, 
                int local_ncols, int local_nrows, MPI_Status status, MPI_Request send_request1, MPI_Request recv_request1, MPI_Request send_request2, MPI_Request recv_request2);
int calc_nrows_from_rank(int rank, int size, int nx);

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

  MPI_Request send_request, recv_request;
  MPI_Request send_request1, recv_request1, send_request2, recv_request2;

  int up;              /* the rank of the process to the left */
  int down;             /* the rank of the process to the right */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

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
  up = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  down = (rank + 1) % size;

  if(rank == MASTER) up = MPI_PROC_NULL;
  if(rank == size - 1) down = MPI_PROC_NULL;

  // Allocate the image
  float* image = (float *)_mm_malloc(sizeof(float) * width * height, 64);

  local_nrows = calc_nrows_from_rank(rank, size, nx);
  local_ncols = ny;

  /* check whether the initialisation was successful */
  if ( local_nrows < 1 )
  {
    fprintf(stderr,"Error: too many processes:- local_nrows < 1\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  int section_nrows = local_nrows + 2;

  float* section = (float *) _mm_malloc(sizeof(float) * section_nrows * (local_ncols + 2), 64);
  float* tmp_section = (float *) _mm_malloc(sizeof(float) * section_nrows * (local_ncols + 2), 64);

  // Set the input image
  init_image(nx, ny, width, height, image);
  
  int chunk = floor(nx/size);

  // Initialising the sections
  for(int i = 0; i < local_nrows + 2; i++) 
  {
    for(int j = 0; j < local_ncols + 2; j++) 
    {
      if (j > 0 && j < (local_ncols + 1) && i > 0 && i < (local_nrows + 1))
      { 
        section[i * (local_ncols + 2) + j] = image[( i * width + j + (chunk * rank * width) )];
        tmp_section[i * (local_ncols + 2) + j] = image[( i * width + j + (chunk * rank * width) )];                 
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
    halo_exchange(section, up, down, local_ncols, local_nrows, status, send_request1, recv_request1, send_request2, recv_request2);

    // Call stencil from section to tmp_section
    stencil(local_nrows, local_ncols, section, tmp_section);

    // Halo Exchange from left to right followed by right to left for tmp_section
    halo_exchange(tmp_section, up, down, local_ncols, local_nrows, status, send_request1, recv_request1, send_request2, recv_request2);

    // Call stencil from tmp_section to section
    stencil(local_nrows, local_ncols, tmp_section, section);
  }
  double toc = wtime();

  // Gathering
  if(rank == MASTER)
  {
    for(int i = 1; i < local_nrows + 1; i++)
    {
      for(int j = 1; j < local_ncols + 1 ; j++)
      {
        image[(i * width) + j] = section[i * (local_ncols + 2) + j];
      }
    }

    for(int r = 1; r < size; r++)
    { 
      int offset = r * local_nrows;       // offset for each rank when storing back to image
      int nrows = calc_nrows_from_rank(r, size, nx);
      for(int i = 1; i < nrows + 1; i++)
      {
        MPI_Irecv(&image[(i + offset) * width + 1], local_ncols, MPI_FLOAT, r, 0, MPI_COMM_WORLD, &recv_request);
        MPI_Wait(&recv_request, &status);
      }
    }
  }
  else
  {
    for(int i = 1; i < local_nrows + 1; i++)
    {
      MPI_Isend(&section[i * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &send_request);
      MPI_Wait(&send_request, &status);
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

  _mm_free(image);
  _mm_free(section);
  _mm_free(tmp_section);
}

void halo_exchange(float * restrict section, int up, int down, int local_ncols, int local_nrows, MPI_Status status, MPI_Request send_request1, MPI_Request recv_request1, MPI_Request send_request2, MPI_Request recv_request2)
{   
    // Sending to up first, then receiving from down
    MPI_Isend(&section[(local_ncols + 2) + 1], local_ncols, MPI_FLOAT, up, 0, MPI_COMM_WORLD, &send_request1);
    MPI_Irecv(&section[(local_nrows + 1) * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, down, 0, MPI_COMM_WORLD, &recv_request1);

    // Sending to down, receiving from up
    MPI_Isend(&section[local_nrows * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &send_request2);
    MPI_Irecv(&section[1], local_ncols, MPI_FLOAT, up, 1, MPI_COMM_WORLD, &recv_request2);
   
    // Waiting
    MPI_Wait(&send_request1, &status);
    MPI_Wait(&recv_request1, &status);

    // Waiting
    MPI_Wait(&send_request2, &status);
    MPI_Wait(&recv_request2, &status);
}

void stencil(const int local_nrows, const int local_ncols,
             float * restrict image, float * restrict tmp_image)
{ 

  // Register variables for iterating through the loops
  register int i;
  register int j;

  for (i = 1; i < local_nrows + 1; ++i)
  {
    for (j = 1; j < local_ncols + 1; ++j)
    { 
      __assume_aligned(image, 64);
      __assume_aligned(tmp_image, 64);
      int cell = j + i * (local_ncols + 2);      
      tmp_image[cell] = ((image[cell] * 6.0f) + (image[cell - (local_ncols + 2)] + image[cell + (local_ncols + 2)] + image[cell - 1] +  image[cell + 1]))/10.0f;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float * restrict image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
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
                  const int width, const int height, float * restrict image)
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

int calc_nrows_from_rank(int rank, int size, int nx)
{
  int nrows;

  nrows = nx / size;       /* integer division */
  if ((nx % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += nx % size;  /* add remainder to last rank */
  }

  return nrows;
}

