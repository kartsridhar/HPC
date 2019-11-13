#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image, int rank, int size);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
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

  // Adding Section stuff
  float *section;         /* section to hold values to send*/
  float *tmp_section;     /* section to hold received values */

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  /* initialise our MPI environment */
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  int ii,jj;             /* row and column indices for the section */

  /* 
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  local_nrows = height;
  local_ncols = calc_ncols_from_rank(rank, size, height);
  /* check whether the initialisation was successful */
  if ( local_ncols < 1 ) {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  section = (float*) malloc(sizeof(float*) * local_nrows * (local_ncols + 2));
  tmp_section = (float*) malloc(sizeof(float*) * local_nrows * (local_ncols + 2));

  // Allocate the image
  float *image = malloc(sizeof(float) * width * height);;
  float *tmp_image = malloc(sizeof(float) * width * height);
  float *gathered = malloc(sizeof(float) * width * height);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  // Adding elements in the sections - INITIALISING
  int section_start = (rank * (width/size)) - 1;
  for(ii = 0; ii < local_nrows; ++ii)
  {
    for(jj = 1; jj < local_ncols + 1; ++jj)
    {
      int cell = ii + jj * height;
      section[cell] = image[ii + (section_start + jj) * height];
      tmp_section[cell] = tmp_image[ii + (section_start + jj) * height];
    }
  }

  // Sending stuff to left of the section and receiving to the right.
  if(rank != MASTER) 
    MPI_Ssend(&section[height], height, MPI_FLOAT, left, 0, MPI_COMM_WORLD);
    
  if(rank != size - 1)
    MPI_Recv(&section[(local_ncols + 1) * height], height, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);
    
  // Sending stuff to the right and receiving to the left
  if(rank != size - 1)
    MPI_Ssend(&section[local_ncols * height], height, MPI_FLOAT, right, 0, MPI_COMM_WORLD);
    
  if(rank != MASTER)
    MPI_Recv(&section[0], height, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);
    
  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {

    // First stencil from section to tmp_section depending on rank
    stencil(local_ncols, ny, width, height, section, tmp_section, rank, size);

    // 1.
    if(rank != MASTER)
      MPI_Send(&tmp_section[height], height, MPI_FLOAT, left, 0, MPI_COMM_WORLD);
          
    if(rank != size - 1)
      MPI_Recv(&tmp_section[(local_ncols + 1) * height], height, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);
      
    // 2.
    if(rank != size - 1)
      MPI_Send(&tmp_section[local_ncols * height], height, MPI_FLOAT, right, 0, MPI_COMM_WORLD);
      
    if(rank != MASTER)
      MPI_Recv(&tmp_section[0], height, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);
      
    // Stencil from tmp_section to section depending on rank
    stencil(local_ncols, ny, width, height, tmp_section, section, rank, size);

    // 1. 
    if(rank != MASTER)
      MPI_Send(&section[height], height, MPI_FLOAT, left, 0, MPI_COMM_WORLD);

    if(rank != size - 1)
      MPI_Recv(&section[(local_ncols + 1) * height], height, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);

    // 2.
    if(rank != size - 1)
      MPI_Send(&section[local_ncols * height], height, MPI_FLOAT, right, 0, MPI_COMM_WORLD);

    if(rank != MASTER)
      MPI_Recv(&section[0], height, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);
  } 
  double toc = wtime();

  // GATHERING ALL THE SECTIONS if rank is MASTER otherwise send all the sections to MASTER
  if(rank == MASTER)
  {
    for(ii = 0; ii < local_nrows; ++ii)
    {
      for(jj = 1; jj < local_ncols + 1; ++jj)
      {
        int cell =  ii + (jj - 1) * height;
        gathered[cell] = section[cell - height];
      }
    }

    // Receiving stuff from each rank into the respective rank
    for(int _rank = 0; _rank < size; ++_rank)
    {
      int section_start = (_rank * (width/size)) - 1;
      int gather_ncols = calc_ncols_from_rank(_rank, size, height);

      for(jj = 1; jj < gather_ncols + 1; ++jj)
      {
        MPI_Recv(&gathered[(section_start + jj) * height], height, MPI_FLOAT, _rank, 0, MPI_COMM_WORLD, &status);
      }
    }
  }
  else  // sending everything to the MASTER
  {
    for(jj = 1; jj < local_ncols + 1; ++jj)
    {
      MPI_Ssend(&section[jj * height], height, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
    }
  }
  

  // Output if rank is MASTER
  if(rank == MASTER)
  {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, width, height, gathered);
  }

  MPI_Finalize();

  free(gathered);
  free(image);
  free(tmp_image);
  free(section);
  free(tmp_section);
}

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image, int rank, int size)
{ 
  if(rank == MASTER)
  {
    // Handling non-edge cases
    for (int i = 2; i < nx + 1; ++i) 
    {
      for (int j = 1; j < ny + 1; ++j) 
      {
        int cell = j + i * height;
        tmp_image[cell] = image[cell] * 0.6f + (image[cell - height] + image[cell + height] + image[cell - 1] +  image[cell + 1]) * 0.1f;      
      }
    }

    // Handling top and bottom rows
    for(int i = 2; i < nx + 1; ++i)
    {
      int top = i * height;
      tmp_image[top] = image[top] * 0.6f + (image[top - height] + image[top + height] + image[top + 1]) * 0.1f;

      int bottom = i * height + (height - 1);
      tmp_image[bottom] = image[bottom] * 0.6f + (image[bottom - height] + image[bottom + height] + image[bottom - 1]) * 0.1f;
    }

    // Handling left-most column
    for(int j = 1; j < ny - 1; j++)
    {
      int left_most = j * height;
      tmp_image[left_most] = image[left_most] * 0.6f + (image[left_most + height] + image[left_most + 1] + image[left_most - 1]) * 0.1f;
    }

    // Handling left-top corner
    tmp_image[height] = image[height] * 0.6f + (image[height + 1] + image[height + height]) * 0.1f;

    // Handling left-bottom corner
    int left_bottom = 2 * height - 1; // height - 1 + height
    tmp_image[left_bottom] = image[left_bottom] * 0.6f + (image[left_bottom - 1] + image[left_bottom + height]) * 0.1f;
  }
  else if(rank == size - 1)  // LAST SECTION
  {
    // Handling non-edge cases
    for(int i = 1; i < nx + 1; ++i)
    {
      for(int j = 1; j < ny - 1; ++j)
      {
        int cell = j + i * height;
        tmp_image[cell] = image[cell] * 0.6f + (image[cell - height] + image[cell + height] + image[cell - 1] + image[cell + 1]) * 0.1f;
      }
    }

    // Handling top and bottom rows
    for(int i = 1; i < nx + 1; ++i)
    {
      int top = i * height;
      tmp_image[top] = image[top] * 0.6f + (image[top - height] + image[top + height] + image[top + 1]) * 0.1f;

      int bottom = i * height + (height - 1);
      tmp_image[bottom] = image[bottom] * 0.6f + (image[bottom - height] + image[bottom + height] + image[bottom - 1]) * 0.1f;
    }

    // Handling right-most column
    for(int j = 1; j < ny - 2; j++)
    {
      int right_most = j + ((width - 2) * height);
      tmp_image[right_most] = image[right_most] * 0.6f + (image[right_most - height] + image[right_most + 1] + image[right_most - 1]) * 0.1f;
    }

    // Handling right-top corner
    int right_top = width * height;
    tmp_image[right_top] = image[right_top] * 0.6f + (image[right_top - height] + image[right_top + 1]) * 0.1f;

    // Handling right-bottom corner
    int right_bottom = ((width + 1) * height) - 1;
    tmp_image[right_bottom] = image[right_bottom] * 0.6f + (image[right_bottom - height] + image[right_bottom - 1]) * 0.1f;
  }
  else // MIDDLE SECTIONS
  {
    // Handling non-edge cases
    for(int i = 1; i < nx + 1; ++i)
    {
      for(int j = 1; j < ny + 1; ++j)
      {
        int cell = j + i * height;
        tmp_image[cell] = image[cell] * 0.6f + (image[cell - height] + image[cell + height] + image[cell- 1] + image[cell + 1]) * 0.1f;
      }
    }

    // Handling the top and bottom rows
    for(int i = 1; i < nx + 1; ++i)
    {
      int top = i * height;
      tmp_image[top] = image[top] * 0.6f + (image[top - height] + image[top + height] + image[top + 1]) * 0.1f;

      int bottom = i * height + (height - 1);
      tmp_image[bottom] = image[bottom] * 0.6f + (image[bottom - height] + image[bottom + height] + image[bottom - 1]) * 0.1f;
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

