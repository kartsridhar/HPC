#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int local_nrows, const int local_ncols, const int width, const int height,
             float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float * restrict image, float * restrict tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float * restrict image);
double wtime(void);

int calc_ncols_from_rank(int rank, int size, int height)
{
  int ncols;

  ncols = height / size;       /* integer division */
  if ((height % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += height % size;  /* add remainder to last rank */
  }
  
  return ncols;
}

int main(int argc, char* argv[])
{
  // Check usage
  if (argc != 4) {
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

  sendbuf = (float*)malloc(sizeof(float) * (local_nrows + 2));
  recvbuf = (float*)malloc(sizeof(float) * (local_nrows + 2));

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
  
  /* check whether the initialisation was successful */
  if ( local_ncols < 1 ) {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  
  int section_ncols = local_ncols + 2;

  printf("section_nrows = %d, section_cols = %d for rank %d\n", local_nrows, section_ncols, rank);

  float* section = malloc(sizeof(float) * (local_nrows + 2) * section_ncols);
  float* tmp_section = malloc(sizeof(float) * (local_nrows + 2) * section_ncols);

  printf("Allocated space for section and tmp_section for rank = %d\n", rank);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  // Initialising the sections
  for(int i = 0; i < local_nrows + 2; i++) {
    for(int j = 0; j < local_ncols + 2; j++) {
      if (j > 0 && j < (local_ncols + 1) && i > 0 && i < (local_nrows + 1)) 
      {
        section[i * (local_ncols + 2) + j] = image[(i * width + j + (ny/size * rank))];
        tmp_section[i * (local_ncols + 2) + j] = image[(i * width + j + (ny/size * rank))];                 /* core cells */
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
  for (int t = 0; t < niters; ++t) {

    for(int i = 0; i < local_nrows + 2; ++i)
      sendbuf[i] = section[i * (local_ncols + 2) + 1];

    MPI_Sendrecv(sendbuf, local_nrows + 2, MPI_FLOAT, left, 0, 
    recvbuf, local_nrows + 2, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);

    // printf("Rank %d performs Send to the LEFT and Receive to the RIGHT successfully\n", rank);

    if(rank != size - 1)
    {
      for(int i = 0; i < local_nrows + 2; ++i)
        section[i * (local_ncols + 2) + local_ncols + 1] = recvbuf[i];
    }

    // SEND right
    for(int i = 0; i < local_nrows + 2; ++i)
      sendbuf[i] = section[i * (local_ncols + 2) + local_ncols];

    MPI_Sendrecv(sendbuf, local_nrows + 2, MPI_FLOAT, right, 0, 
    recvbuf, local_nrows + 2, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);

    // printf("Rank %d performs Send to the RIGHT and Receive to the LEFT successfully\n", rank);

    if(rank != MASTER)
    {
      for(int i = 0; i < local_nrows + 2; ++i)
        section[i * (local_ncols + 2)] = recvbuf[i];
    }

    stencil(local_ncols, local_nrows, width, height, section, tmp_section);    
    // printf("Applied stencil from section to tmp_section for rank %d\n", rank);

    for(int i = 0; i < local_nrows + 2; ++i)
      sendbuf[i] = tmp_section[i * (local_ncols + 2) + 1];

    MPI_Sendrecv(sendbuf, local_nrows + 2, MPI_FLOAT, left, 0, 
    recvbuf, local_nrows + 2, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);

    // printf("Rank %d performs Send to the LEFT and receive to the RIGHT successfully\n", rank);

    if(rank != size - 1)
    {
      for(int i = 0; i < local_nrows + 2; ++i)
        tmp_section[i * (local_ncols + 2) + local_ncols + 1] = recvbuf[i];
    }

    // SEND right
    for(int i = 0; i < local_nrows + 2; ++i)
      sendbuf[i] = tmp_section[i * (local_ncols + 2) + local_ncols];

    MPI_Sendrecv(sendbuf, local_nrows + 2, MPI_FLOAT, right, 0, 
    recvbuf, local_nrows + 2, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);

    // printf("Rank %d performs Send to the RIGHT and receive to the LEFT successfully\n", rank);

    if(rank != MASTER)
    {
      for(int i = 0; i < local_nrows + 2; ++i)
        tmp_section[i * (local_ncols + 2)] = recvbuf[i];
    }

    stencil(local_ncols, local_nrows, width, height, tmp_section, section);
    // printf("Applied stencil from tmp_section to section for rank %d\n", rank);
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

        // offset for each rank when storing back to image
        int offset = r * (ny / size) + 1;

        MPI_Recv(&image[(i * width) + offset], local_ncols , MPI_FLOAT, r, 0, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      MPI_Send(&section[i * (local_ncols + 2) + 1], local_ncols , MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
    }
  }
  
  printf("Final Image added from Rank %d\n", rank);

  // Output if rank is MASTER
  if(rank == MASTER)
  {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }

  free(image);
  free(tmp_image);
  free(sendbuf);
  free(recvbuf);
  free(section);
  free(tmp_section);

  MPI_Finalize();
}

void stencil(const int local_nrows, const int local_ncols, const int width, const int height,
             float * restrict image, float * restrict tmp_image)
{ 
  for (int i = 1; i < local_nrows + 1; ++i)
  {
    for (int j = 1; j < local_ncols + 1; ++j) 
    {
      int cell = j + i * (local_nrows + 2);
      tmp_image[cell] = image[cell] * 0.6f + (image[cell - (local_nrows + 2)] + image[cell + (local_nrows + 2)] + image[cell - 1] +  image[cell + 1]) * 0.1f;      
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float * restrict image, float * restrict tmp_image)
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
