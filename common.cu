

bool check(uint8_t* a, uint8_t* b, int num) {
  return memcmp(a, b, num * sizeof(uint8_t));
}


void fprint_uint8_array(FILE* stream, uint8_t* array, int size) {
    for (int i = 0; i < size; i ++) {
        fprintf(stream, "%02x", array[i]);
    }
    fprintf(stream, "\n");
}
