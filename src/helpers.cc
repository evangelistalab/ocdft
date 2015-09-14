#include "helpers.h"

namespace psi{

void extract_square_subblock(SharedMatrix A, SharedMatrix B, bool occupied, Dimension npi, double diagonal_shift)
{
    // Set the diagonal of B to diagonal_shift
    B->identity();
    B->scale(diagonal_shift);

    int nirrep_ = A->nirrep();
    Dimension nmopi_ = A->colspi();

    // Copy the block from A
    for (int h = 0; h < nirrep_; ++h){
        int block_dim = occupied ? npi[h] : nmopi_[h] - npi[h];
        int block_shift = occupied ? 0 : npi[h];
        if (block_dim != 0){
            double** A_h = A->pointer(h);
            double** B_h = B->pointer(h);
            for (int i = 0; i < block_dim; ++i){
                for (int j = 0; j < block_dim; ++j){
                    B_h[i][j] = A_h[i + block_shift][j + block_shift];
                }
            }
        }
    }
}

void copy_subblock(SharedMatrix A, SharedMatrix B, Dimension rowspi, Dimension colspi, bool occupied)
{
    int nirrep_ = A->nirrep();
    Dimension nmopi_ = A->colspi();
    for (int h = 0; h < nirrep_; ++h){
        int nrows = occupied ? rowspi[h] : nmopi_[h] - rowspi[h];
        int row_shift = occupied ? 0 : rowspi[h];
        int ncols = occupied ? colspi[h] : nmopi_[h] - colspi[h];
        int col_shift = occupied ? 0 : colspi[h];
        if (nrows * ncols != 0){
            double** A_h = A->pointer(h);
            double** B_h = B->pointer(h);
            for (int i = 0; i < nrows; ++i){
                for (int j = 0; j < ncols; ++j){
                    B_h[i][j] = A_h[i + row_shift][j + col_shift];
                }
            }
        }
    }
}

void copy_block(SharedMatrix A, double alpha, SharedMatrix B, double beta, Dimension rowspi, Dimension colspi,
                      Dimension A_rows_offsetpi, Dimension A_cols_offsetpi,
                      Dimension B_rows_offsetpi, Dimension B_cols_offsetpi)
{
    int nirrep_ = A->nirrep();
    Dimension nmopi_ = A->colspi();
    for (int h = 0; h < nirrep_; ++h){
        int nrows = rowspi[h];
        int ncols = colspi[h];
        int A_row_offset = A_rows_offsetpi[h];
        int A_col_offset = A_cols_offsetpi[h];
        int B_row_offset = B_rows_offsetpi[h];
        int B_col_offset = B_cols_offsetpi[h];
        if (nrows * ncols != 0){
            double** A_h = A->pointer(h);
            double** B_h = B->pointer(h);
            for (int i = 0; i < nrows; ++i){
                for (int j = 0; j < ncols; ++j){
                    B_h[i + B_row_offset][j + B_col_offset] = alpha * A_h[i + A_row_offset][j + A_col_offset] + beta * B_h[i + B_row_offset][j + B_col_offset];
                }
            }
        }
    }
}

}
