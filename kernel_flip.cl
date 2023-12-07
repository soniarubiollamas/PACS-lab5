__kernel void image_flip(__global unsigned char *image_data_in,
                         __global unsigned char *image_data_out,
                         const int width,
                         const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width / 2) {
        int source_index = y * width + x;
        int target_index = y * width + (width - 1 - x);

        unsigned char temp = image_data_in[source_index];
        image_data_out[source_index] = image_data_in[target_index];
        image_data_out[target_index] = temp;
    }
}