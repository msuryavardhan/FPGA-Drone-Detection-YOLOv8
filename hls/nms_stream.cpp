#include <hls_stream.h>
#include <ap_int.h>

#define MAX_BOXES 8400

// Packed bounding box (80 bits total)
typedef ap_uint<80> bbox_pkt;

// Function prototype
void nms_stream(
    hls::stream<bbox_pkt> &in_stream,
    hls::stream<bbox_pkt> &out_stream,
    int num_boxes,
    ap_uint<8> conf_threshold
) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=num_boxes bundle=CTRL
#pragma HLS INTERFACE s_axilite port=conf_threshold bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

#pragma HLS PIPELINE II=1

    for (int i = 0; i < num_boxes; i++) {

        bbox_pkt pkt = in_stream.read();

        // Extract confidence (last 8 bits)
        ap_uint<8> conf = pkt.range(7,0);

        if (conf >= conf_threshold) {
            out_stream.write(pkt);
        }
    }
}
