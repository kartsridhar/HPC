#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0
#define NROWS 1024       // number of rows in the image
#define NCOLS 1024/4     // number of cols in the image
#define SECTION_SIZE ((NROWS * NCOLS) / 4);

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image, int rank);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);

int calc_ncols_from_rank(int rank, int size)
{
  int ncols;

  ncols = NCOLS / size;       /* integer division */
  if ((NCOLS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += NCOLS % size;  /* add remainder to last rank */
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
  char message[BUFSIZ];

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

  local_nrows = NROWS;
  local_ncols = calc_ncols_from_rank(rank, size);
  /* check whether the initialisation was successful */
  if ( local_ncols < 1 ) {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  // Allocate the image
  float* image;
  float* tmp_image;

  if(rank == MASTER)
  {
    image = malloc(sizeof(double) * width * height);
    tmp_image = malloc(sizeof(double) * width * height);

    // Set the input image
    init_image(nx, ny, width, height, image, tmp_image);
  }

  float *buffer = malloc(sizeof(double) * SECTION_SIZE);
  float *tmp_buffer = malloc(sizeof(double) * SECTION_SIZE);

  // Send data from MASTER to all buffer
  MPI_Scatter(image, SECTION_SIZE, MPI_FLOAT, buffer, SECTION_SIZE, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  // Call the stencil kernel
  double tic = wtime();
  
  for (int t = 0; t < niters; ++t) {
    stencil(nx/size, ny, buffer, tmp_buffer, rank);
    stencil(nx/size, ny, tmp_buffer, buffer, rank);
  }
  
  double toc = wtime();

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc - tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, width, height, image);
  
  free(buffer);
  free(tmp_buffer);
  free(image);
  free(tmp_image);

  MPI_Finalize();
}

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image, int rank)
{
    /* 
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  int left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  int right = (rank + 1) % size;

  // checking if master
  // sending last row of the array to worker with rank 1
  if(rank == MASTER)
  {

  }
  for (int i = 1; i < nx + 1; ++i) {
    for (int j = 1; j < ny + 1; ++j) {
       int cell = j + i * height;
       tmp_image[cell] = image[cell] * 0.6f + (image[cell - height] + image[cell + height] + image[cell - 1] +  image[cell + 1]) * 0.1f;      
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

