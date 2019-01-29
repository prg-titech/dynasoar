// Code adapted from:
// https://stackoverflow.com/questions/8126815/how-to-read-in-data-from-a-pgm-file-in-c

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "dataset_loader.h"

using namespace std;

dataset_t load_from_file(char* filename) {
  int row = 0, col = 0, numrows /*y*/ = 0, numcols /*x*/ = 0;
  ifstream infile(filename);
  stringstream ss;
  string inputLine = "";

  // First line : version
  getline(infile,inputLine);
  if(inputLine.compare("P2") != 0) cerr << "Version error" << endl;
  else {
#ifndef NDEBUG
    cout << "Version : " << inputLine << endl;
#endif  // NDEBUG
  }

  // Second line : comment
  getline(infile,inputLine);
#ifndef NDEBUG
  cout << "Comment : " << inputLine << endl;
#endif  // NDEBUG

  // Continue with a stringstream
  ss << infile.rdbuf();
  // Third line : size
  ss >> numcols >> numrows;
#ifndef NDEBUG
  cout << numcols << " columns and " << numrows << " rows" << endl;
#endif  // NDEBUG

  vector<int>* alive_cells = new vector<int>();

  // Following lines : data
  for(row = 0; row < numrows; ++row) {
    for (col = 0; col < numcols; ++col) {
      int pixel;
      ss >> pixel;

      if (pixel == 0) {
        // Cell is alive.
        alive_cells->push_back(col + row*numcols);
      }
    }
  }

  infile.close();
#ifndef NDEBUG
  cout << "Loaded dataset from file." << endl;
#endif  // NDEBUG

  return dataset_t(/*x=*/ numcols, /*y=*/ numrows,
                   /*alive_cells=*/ alive_cells->data(),
                   /*num_alive=*/ alive_cells->size());
}


dataset_t load_glider() {
  dataset_t result;
  result.x = 200;
  result.y = 100;

  // Create data set.
  int* cell_ids = new int[5];
  cell_ids[0] = 1 + 0*result.x;
  cell_ids[1] = 2 + 1*result.x;
  cell_ids[2] = 0 + 2*result.x;
  cell_ids[3] = 1 + 2*result.x;
  cell_ids[4] = 2 + 2*result.x;
  result.alive_cells = cell_ids;
  result.num_alive = 5;

  return result;
}
