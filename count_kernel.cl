__kernel void count_kernel(__global int* output) {
    int id = get_global_id(0);
    output[id] = id + 1;
}