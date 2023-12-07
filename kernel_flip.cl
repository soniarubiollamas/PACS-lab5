__kernel void image_flip(__global unsigned char *image_data,
                         const int width,
                         const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width / 2) {
        int source_index = y * width + x;
        int target_index = y * width + (width - 1 - x);

        unsigned char temp = image_data[source_index];
        image_data[source_index] = image_data[target_index];
        image_data[target_index] = temp;
    }
}
