kernel void pref_sum(global const float *arr, global float *result, uint N) {
    const uint id = get_local_id(0);
    const uint part_len = N / TS;

    local float buffer_1[TS];
    local float buffer_2[TS];

    float sum = 0.0f;

    for (size_t t = 0; t < part_len; ++t) {
        buffer_1[id] = arr[t * TS + id];

        barrier(CLK_LOCAL_MEM_FENCE);

        bool last = false;

        for (size_t i = 1, t = 0; i < TS; i <<= 1, t++) {
            if (t & 1) {
                if (id < i) {
                    buffer_1[id] = buffer_2[id];
                } else {
                    buffer_1[id] = buffer_2[id] + buffer_2[id - i];
                }
                last = false;
            } else {
                if(id < i){
                    buffer_2[id] = buffer_1[id];
                } else {
                    buffer_2[id] = buffer_1[id] + buffer_1[id - i];
                }
                last = true;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[t * TS + id] = sum;
        if (last) {
            result[t * TS + id] += buffer_2[id];
            sum += buffer_2[TS - 1];
        } else {
            result[t * TS + id] += buffer_1[id];
            sum += buffer_1[TS - 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
