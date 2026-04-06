#pragma once

int htp_ops_rpc_rms_norm_f32(int dst_fd, int dst_offset, int src_fd, int src_offset, int ne0, int ne1);

int htp_ops_rpc_mat_mul_af32_pwf16_of32(int output_fd, int output_offset, int activation_fd, int activation_offset,
                                        int weight_fd, int weight_offset, int m, int k, int n);
int htp_ops_rpc_mat_mul_af32_wf16_of32(int output_fd, int output_offset, int activation_fd, int activation_offset,
                                        int weight_fd, int weight_offset, int m, int k, int n);

int htp_ops_rpc_mat_mul_af32_pwqk0_of32(int output_fd, int output_offset, int activation_fd, int activation_offset,
                                        int weight_fd, int weight_offset, int m, int k, int n, int weight_type);
