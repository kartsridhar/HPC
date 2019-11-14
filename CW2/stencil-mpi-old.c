#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height,
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
  float * restrict image = malloc(sizeof(float) * width * height);;
  float * restrict tmp_image = malloc(sizeof(float) * width * height);
  
  local_nrows = height;
  local_ncols = calc_ncols_from_rank(rank, size, ny);
  
  /* check whether the initialisation was successful */
  if ( local_ncols < 1 ) {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  
  int section_ncols = local_ncols + 2;

  printf("section_nrows = %d, section_cols = %d for rank %d\n", local_nrows, section_ncols, rank);

  float * restrict section = malloc(sizeof(float) * local_nrows * section_ncols);
  float * restrict tmp_section = malloc(sizeof(float) * local_nrows * section_ncols);

  printf("Allocated space for section and tmp_section for rank = %d\n", rank);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  // Initialising the sections
  if(rank == MASTER)
  { 
    for(int i = 0; i < local_nrows; ++i)
    {
      for(int j = 1; j < section_ncols; ++j)
      {
        section[j + i * local_nrows] = image[(j - 1) + i * local_nrows];
        tmp_section[j + i * local_nrows] = image[(j - 1) + i * local_nrows];
      }

    }
    printf("Sections for MASTER initialised successfully\n");
  }
  else
  {
    int section_start = rank * local_nrows * local_ncols + local_nrows;
    for(int i = 0; i < local_nrows; ++i)
    { 
      for(int j = 1; j < section_ncols; ++j)
      {
        section[j + i * local_nrows] = image[(j - 1) + i * local_nrows];
        tmp_section[j + i * local_nrows] = image[(j - 1) + i * local_nrows];
      }
    }
    printf("Sections for rank %d initialised successfully\n", rank);
  }

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {

    if(rank != MASTER) 
    {
      MPI_Sendrecv(&tmp_section[local_nrows], local_nrows, MPI_FLOAT, left, 0, 
      &tmp_section[0], local_nrows, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);

      printf("Rank %d performs Send and Receive to the LEFT successfully\n", rank);
    }
      
    if(rank != size - 1)
    {
      MPI_Sendrecv(&tmp_section[local_nrows * section_ncols - (2 * local_nrows)], local_nrows, MPI_FLOAT, right, 0, 
      &tmp_section[local_nrows * section_ncols - local_nrows], local_nrows, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);

      printf("Rank %d performs Send and Receive to the RIGHT successfully\n", rank);
    }

    //stencil(section_ncols, local_nrows, width, height, section, tmp_section);

    printf("Applied stencil from section to tmp_section for rank %d\n", rank);

    if(rank != MASTER)
    {
      MPI_Sendrecv(&section[local_nrows], local_nrows, MPI_FLOAT, left, 0, 
      &section[0], local_nrows, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);

      printf("Rank %d performs Send and Receive to the LEFT successfully\n", rank);
    }

    if(rank != size - 1)
    {
      MPI_Sendrecv(&section[local_nrows * section_ncols - (2 * local_nrows)], local_nrows, MPI_FLOAT, right, 0, 
      &section[local_nrows * section_ncols - local_nrows], local_nrows, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);

      printf("Rank %d performs Send and Receive to the RIGHT successfully\n", rank);
    }

    //stencil(section_ncols, local_nrows, width, height, tmp_section, section);

    printf("Applied stencil from tmp_section to section for rank %d\n", rank);
  } 
  double toc = wtime();

  if(rank == MASTER)
  {
    for(int i = 0; i < (local_nrows * local_ncols); ++i)
    {
      image[i] = section[i];
    }

    for(int _rank = 1; _rank < size - 1; ++_rank)
    {
      int section_start = _rank * (local_nrows * local_ncols);
      for(int j = 0; j < local_ncols; ++j)
      {
        MPI_Recv(&image[section_start + j * local_nrows], local_nrows, MPI_FLOAT, _rank, 0, MPI_COMM_WORLD, &status);
      }
    }

    for(int last = 0; last < ny - ((size - 1) * local_ncols); ++last)
    {
      MPI_Recv(&image[(size - 1) * (local_nrows * local_ncols) + last * local_nrows], local_nrows, MPI_FLOAT, size - 1, 0, MPI_COMM_WORLD, &status);
    }
  }
  else
  {
    for(int i = 1; i < section_ncols; ++i)
    {
      MPI_Send(&section[i * local_nrows], local_nrows, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
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
  // free(tmp_image);
  free(section);
  free(tmp_section);

  MPI_Finalize();
}

void stencil(const int nx, const int ny, const int width, const int height,
             float * restrict image, float * restrict tmp_image)
{ 
  printf("nx = %d, ny = %d, width = %d, height = %d \n", nx, ny, width, height);
  // Middle Section
  for(int i = 0; i < nx; ++i)
  {
    for(int j = 1; j < ny + 1; ++j)
    { 
      int index = j + i * height;

      tmp_image[index] = image[index] * 0.6f + (image[index - 1] + image[index + 1] + image[index - height] + image[index + height]) * 0.1f;
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