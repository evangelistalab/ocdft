#ifndef _noci_h_
#define _noci_h_

#include "libmints/matrix.h"

namespace psi{

// Helper functions
/// Extract a block from matrix A and copies it to B
void extract_square_subblock(SharedMatrix A, SharedMatrix B, bool occupied, Dimension npi, double diagonal_shift);

/// Copy a subblock of dimension rowspi x colspi from matrix A into B.  If desired, it can copy the complementary subblock
void copy_subblock(SharedMatrix A, SharedMatrix B, Dimension rowspi, Dimension colspi,bool occupied);

/// Copy a subblock of dimension rowspi x colspi from matrix A into B.  If desired, it can copy the complementary subblock
void copy_block(SharedMatrix A, double alpha, SharedMatrix B, double beta, Dimension rowspi, Dimension colspi,
                Dimension A_rows_offsetpi = Dimension(8), Dimension A_cols_offsetpi = Dimension(8),
                Dimension B_rows_offsetpi = Dimension(8), Dimension B_cols_offsetpi = Dimension(8));
}

#endif // _noci_h_
