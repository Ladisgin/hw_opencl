kernel void matrix_mul(global const float *a, global const float *b, global float *c, uint N, uint K, uint M) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int i = get_global_id(0); // 0..N
    const int j = get_global_id(1); // 0..M

    local float local_a[TS][TS];
    local float local_b[TS][TS];

    float t = 0;

    const int tile = K / TS;

    for (int y = 0; y < tile; y++) {
        const int tiled_row = TS * y + row;
        const int tiled_col = TS * y + col;
        local_a[col][row] = a[i * K + tiled_col];
        local_b[col][row] = b[tiled_row * M + j];

        for (uint k = 0; k < TS; k++) {
            t += local_a[k][row] * local_b[col][k];
        }
    }
    c[i * M + j] = t;
}
