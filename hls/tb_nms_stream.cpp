#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>

typedef ap_uint<80> bbox_pkt;

// Function declaration
void nms_stream(
    hls::stream<bbox_pkt> &in_stream,
    hls::stream<bbox_pkt> &out_stream,
    int num_boxes,
    ap_uint<8> conf_threshold
);

int main() {

    hls::stream<bbox_pkt> in_stream;
    hls::stream<bbox_pkt> out_stream;

    // Create 3 test packets
    bbox_pkt pkt1 = 0;
    bbox_pkt pkt2 = 0;
    bbox_pkt pkt3 = 0;

    // Pack data: lower 8 bits = confidence
    pkt1.range(7,0)   = 30;
    pkt2.range(7,0)   = 80;
    pkt3.range(7,0)   = 120;

    in_stream.write(pkt1);
    in_stream.write(pkt2);
    in_stream.write(pkt3);

    int num_boxes = 3;
    ap_uint<8> threshold = 102; // 0.4 * 255 = 102

    nms_stream(in_stream, out_stream, num_boxes, threshold);

    std::cout << "Output boxes:\n";

    while (!out_stream.empty()) {
        bbox_pkt out = out_stream.read();
        std::cout << "Confidence: "
                  << (unsigned int)out.range(7,0)
                  << std::endl;
    }
    int count = 0;

    while (!out_stream.empty()) {
        bbox_pkt out = out_stream.read();
        count++;
    }

    printf("Number of valid boxes = %d\n", count);

    return 0;
}
